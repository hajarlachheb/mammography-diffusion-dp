#!/usr/bin/env python3
"""
Evaluate a trained classifier checkpoint
"""

import argparse
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
import numpy as np
import logging
from datetime import datetime
from PIL import Image, ImageOps

# Matplotlib for ROC curves
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. ROC curves will not be generated.")

# DINOv3 support via timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ResizeContainPad:
    """Resize without changing aspect ratio, then center on a square canvas."""

    def __init__(self, size, fill=0):
        self.size = (int(size), int(size))
        self.fill = fill

    def __call__(self, image):
        contained = ImageOps.contain(image, self.size, method=Image.Resampling.LANCZOS)
        canvas = Image.new(image.mode, self.size, color=self.fill)
        left = (self.size[0] - contained.width) // 2
        top = (self.size[1] - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas


def get_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def clean_state_dict_keys(state_dict):
    """Strip wrappers such as Opacus GradSampleModule (_module.) from keys."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("_module.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_model(checkpoint_path, model_type='resnet50', device='cuda'):
    """Load model from checkpoint. Handles dropout heads added during training."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dropout_prob = (
        float(checkpoint.get("dropout", 0.3))
        if isinstance(checkpoint, dict)
        else 0.3
    )
    
    # Create model with the same head structure as training
    if model_type == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(model.fc.in_features, 2)
        )
    elif model_type == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(model.fc.in_features, 2)
        )
    elif model_type == 'swin_t':
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        # Checkpoints from --head_type ref_mlp record it; older checkpoints have no key and use the linear head.
        head_type = checkpoint.get("head_type", "linear") if isinstance(checkpoint, dict) else "linear"
        if head_type == "ref_mlp":
            model.head = nn.Sequential(
                nn.Linear(in_features, 384),
                nn.LeakyReLU(0.2),
                nn.Linear(384, 2)
            )
        elif head_type == "linear":
            model.head = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(in_features, 2)
            )
        else:
            raise ValueError(f"Unknown head_type in checkpoint: {head_type}")
    elif model_type == 'swin_s':
        model = models.swin_s(weights=None)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 2)
        )
    elif model_type in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for DINOv3 models. Install with: pip install timm")
        model = timm.create_model(model_type, pretrained=False, num_classes=0)  # num_classes=0 removes head
        embedding_dim = model.num_features if hasattr(model, 'num_features') else model.embed_dim
        model.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_dim, 2)
        )
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    # Load weights. DP checkpoints may have been saved from an Opacus wrapper
    # with "_module." prefixes; strip them explicitly and fail if most weights
    # did not load, instead of silently evaluating a near-random model.
    state_dict = clean_state_dict_keys(get_state_dict_from_checkpoint(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Opacus-compatible Swin training removes relative_position_index buffers
    # from each attention block. They are deterministic architecture constants
    # and are recreated correctly by torchvision when the model is constructed.
    allowed_missing = [
        key for key in missing if key.endswith("relative_position_index")
    ]
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if allowed_missing:
        print(
            "Checkpoint load note: restored "
            f"{len(allowed_missing)} deterministic Swin relative-position buffers "
            "from the model architecture"
        )
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint is incompatible with {model_type}: "
            f"missing learned/state keys={disallowed_missing[:20]}, "
            f"unexpected keys={unexpected[:20]}"
        )
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def infer_positive_class_index(class_to_idx: dict) -> int:
    """
    Infer which class index should be treated as the *positive* class for AUROC/AUPRC.

    Convention in this repo's datasets:
      - is_benign_false == malignant
      - is_benign_true  == benign

    torchvision.datasets.ImageFolder assigns indices alphabetically by folder name, so
    relying on a fixed column (e.g., probs[:,1]) is unsafe.
    """
    if not class_to_idx:
        return 1

    # Prefer explicit malignant naming conventions.
    for key in ("is_benign_false", "malignant", "cancer", "positive", "pos"):
        if key in class_to_idx:
            return int(class_to_idx[key])

    # Fallback: assume binary with class index 1 as positive.
    return 1


def get_class_index(class_to_idx: dict, positive_class: str) -> int:
    """
    Resolve the requested positive class to the ImageFolder class index.

    In this project:
      - malignant == is_benign_false
      - benign    == is_benign_true

    Use positive_class='benign' when comparing to Richard/mammo_dp style
    is_benign-positive reporting. Use 'malignant' for clinical reporting.
    """
    if positive_class == "malignant":
        for key in ("is_benign_false", "malignant", "cancer", "positive", "pos"):
            if key in class_to_idx:
                return int(class_to_idx[key])
        return infer_positive_class_index(class_to_idx)

    if positive_class == "benign":
        for key in ("is_benign_true", "benign", "negative", "neg"):
            if key in class_to_idx:
                return int(class_to_idx[key])
        if len(class_to_idx) == 2:
            malignant_idx = infer_positive_class_index(class_to_idx)
            return int([idx for idx in class_to_idx.values() if idx != malignant_idx][0])
        return 1

    raise ValueError(f"Unsupported positive_class: {positive_class}")


def evaluate_model(model, dataloader, device, positive_class_index: int, criterion=None):
    """Evaluate model and return metrics (matching mammo_dp style)"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_positive_probs = []
    all_losses = []
    raw_labels = []
    raw_preds = []
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            all_losses.append(loss.item())
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            positive_probs_np = probs[:, positive_class_index].cpu().numpy()
            binary_preds_np = (preds_np == positive_class_index).astype(int)
            binary_labels_np = (labels_np == positive_class_index).astype(int)

            raw_preds.extend(preds_np)
            raw_labels.extend(labels_np)
            all_preds.extend(binary_preds_np)
            all_labels.extend(binary_labels_np)
            all_probs.extend(positive_probs_np)
            all_positive_probs.extend(positive_probs_np)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_positive_probs = np.array(all_positive_probs)
    raw_labels = np.array(raw_labels)
    raw_preds = np.array(raw_preds)
    
    # Calculate mean loss
    mean_loss = np.mean(all_losses) if all_losses else None
    
    # Calculate metrics (matching mammo_dp style)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        logging.warning(f"Cannot calculate ROC-AUC: {e}")
        roc_auc = None
    
    # PRC-AUC (Average Precision) - matching mammo_dp
    try:
        prc_auc = average_precision_score(all_labels, all_probs)
    except ValueError as e:
        logging.warning(f"Cannot calculate PRC-AUC: {e}")
        prc_auc = None
    
    # Precision, Recall, F1 (for positive class, matching mammo_dp)
    try:
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    except:
        prec, rec, f1 = 0.0, 0.0, 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Handle edge cases where only one class is present
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        unique_labels = set(all_labels)
        unique_preds = set(all_preds)
        if 1 in unique_labels and 1 in unique_preds:
            tp = cm[0, 0]
            tn, fp, fn = 0, 0, 0
        else:
            tn = cm[0, 0]
            tp, fp, fn = 0, 0, 0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        unique_labels = list(set(all_labels))
        unique_preds = list(set(all_preds))
        
        if len(unique_labels) == 1 and len(unique_preds) == 2:
            if unique_labels[0] == 1:
                tp = sum(1 for p in all_preds if p == 1)
                fn = sum(1 for p in all_preds if p == 0)
            else:
                tn = sum(1 for p in all_preds if p == 0)
                fp = sum(1 for p in all_preds if p == 1)
        elif len(unique_labels) == 2 and len(unique_preds) == 1:
            if unique_preds[0] == 1:
                tp = sum(1 for l in all_labels if l == 1)
                fp = sum(1 for l in all_labels if l == 0)
            else:
                tn = sum(1 for l in all_labels if l == 0)
                fn = sum(1 for l in all_labels if l == 1)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Debug: print raw ImageFolder labels and the binary labels used for metrics.
    print(f"  Raw labels distribution: {dict(zip(*np.unique(raw_labels, return_counts=True)))}")
    print(f"  Raw predictions distribution: {dict(zip(*np.unique(raw_preds, return_counts=True)))}")
    print(f"  Binary labels distribution (positive={positive_class_index}): {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    print(f"  Binary predictions distribution (positive={positive_class_index}): {dict(zip(*np.unique(all_preds, return_counts=True)))}")
    
    return {
        'loss': float(mean_loss) if mean_loss is not None else None,
        'accuracy': float(accuracy),
        'auc_roc': float(roc_auc) if roc_auc is not None else None,
        'auc_prc': float(prc_auc) if prc_auc is not None else None,  # PRC-AUC matching mammo_dp
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        # Store full data for ROC curve plotting
        '_y_true': all_labels,
        '_y_prob': all_positive_probs,
        '_binary_predictions': all_preds,
        '_raw_labels': raw_labels,
        '_raw_predictions': raw_preds,
        'positive_class_index': int(positive_class_index),
    }


def load_manifest_metadata(dataset_root, manifest_path=None):
    manifest_path = Path(manifest_path) if manifest_path else Path(dataset_root) / "manifest.csv"
    if not manifest_path.exists():
        return {}, None
    metadata = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path_value = (
                row.get("output_path")
                or row.get("output_image")
                or row.get("path")
                or row.get("source_image")
                or ""
            )
            if not path_value:
                continue
            path = Path(path_value)
            metadata[str(path)] = row
            metadata[path.name] = row
    return metadata, manifest_path


def binary_summary(labels, probabilities, predictions):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    result = {
        "n_images": int(len(labels)),
        "n_negative": int(np.sum(labels == 0)),
        "n_positive": int(np.sum(labels == 1)),
        "accuracy": float(accuracy_score(labels, predictions)) if len(labels) else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if len(labels) and len(np.unique(labels)) == 2
            else None
        ),
        "auc_roc": None,
        "auc_prc": None,
        "sensitivity": None,
        "specificity": None,
    }
    if len(np.unique(labels)) == 2:
        result["auc_roc"] = float(roc_auc_score(labels, probabilities))
        result["auc_prc"] = float(average_precision_score(labels, probabilities))
        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
        result["sensitivity"] = float(tp / (tp + fn)) if tp + fn else None
        result["specificity"] = float(tn / (tn + fp)) if tn + fp else None
    return result


def stratified_summaries(prediction_rows, group_field):
    grouped = {}
    group_values = sorted(
        {
            str(row.get(group_field, "")).strip()
            for row in prediction_rows
            if str(row.get(group_field, "")).strip()
        }
    )
    for value in group_values:
        rows = [row for row in prediction_rows if str(row.get(group_field, "")).strip() == value]
        grouped[value] = binary_summary(
            [row["malignant_label"] for row in rows],
            [row["malignant_probability"] for row in rows],
            [row["malignant_prediction"] for row in rows],
        )
    return grouped


def patient_cluster_bootstrap(prediction_rows, iterations, seed):
    """Bootstrap patients while retaining all images from each sampled patient."""
    by_patient = {}
    for row in prediction_rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if patient_id:
            by_patient.setdefault(patient_id, []).append(row)
    patient_ids = sorted(by_patient)
    if len(patient_ids) < 2 or iterations <= 0:
        return None

    rng = np.random.default_rng(seed)
    aurocs = []
    auprcs = []
    max_attempts = max(iterations * 10, 100)
    attempts = 0
    while len(aurocs) < iterations and attempts < max_attempts:
        attempts += 1
        sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sampled_rows = [
            row
            for patient_id in sampled_ids
            for row in by_patient[str(patient_id)]
        ]
        labels = np.asarray(
            [row["malignant_label"] for row in sampled_rows],
            dtype=int,
        )
        if len(np.unique(labels)) < 2:
            continue
        probabilities = np.asarray(
            [row["malignant_probability"] for row in sampled_rows],
            dtype=float,
        )
        aurocs.append(float(roc_auc_score(labels, probabilities)))
        auprcs.append(float(average_precision_score(labels, probabilities)))

    if not aurocs:
        return None

    def interval(values):
        return {
            "mean": float(np.mean(values)),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }

    return {
        "method": "patient-cluster bootstrap of image-level predictions",
        "n_patients": len(patient_ids),
        "requested_iterations": int(iterations),
        "successful_iterations": len(aurocs),
        "seed": int(seed),
        "auc_roc": interval(aurocs),
        "auc_prc": interval(auprcs),
    }


def build_prediction_rows(dataset, manifest_metadata, arrays):
    paths = [path for path, _ in dataset.samples]
    if len(paths) != len(arrays["raw_labels"]):
        raise RuntimeError(
            f"Prediction/path count mismatch: {len(arrays['raw_labels'])} predictions "
            f"for {len(paths)} files"
        )
    rows = []
    for index, path_value in enumerate(paths):
        path = Path(path_value)
        metadata = (
            manifest_metadata.get(str(path))
            or manifest_metadata.get(path.name)
            or {}
        )
        raw_label = int(arrays["raw_labels"][index])
        raw_prediction = int(arrays["raw_predictions"][index])
        rows.append(
            {
                "index": index,
                "path": str(path),
                "filename": path.name,
                "class_index": raw_label,
                "class_name": dataset.classes[raw_label],
                "predicted_class_index": raw_prediction,
                "predicted_class_name": dataset.classes[raw_prediction],
                "malignant_label": int(arrays["binary_labels"][index]),
                "malignant_probability": float(arrays["probabilities"][index]),
                "malignant_prediction": int(arrays["binary_predictions"][index]),
                "patient_id": str(metadata.get("patient_id", "")),
                "center": str(metadata.get("dataset", metadata.get("center", ""))),
                "view": str(metadata.get("view", "")).upper(),
                "laterality": str(metadata.get("laterality", "")).upper(),
                "sop_instance_uid": str(metadata.get("sop_instance_uid", "")),
            }
        )
    return rows


def write_prediction_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def output_ROC_curve(y_true, y_prob, output_path, description=''):
    """Output ROC curve (matching mammo_dp style)"""
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available. Skipping ROC curve generation.")
        return
    
    if len(np.unique(y_true)) < 2:
        logging.warning("Cannot generate ROC curve: only one class present in labels.")
        return
    
    # y_prob is already the probability of the configured positive class.
    y_prob_pos = y_prob
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {description}')
    plt.legend(loc='lower right')
    
    # Save figure
    roc_path = Path(output_path).parent / f"ROC_curve_{description}.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved to: {roc_path}")

def write_results_txt(output_path, checkpoint_path, dataset_name, metrics, epoch=None):
    """Write results to text file (matching mammo_dp style)"""
    results_file = Path(output_path).parent / 'results_test.txt'
    
    with open(results_file, 'a') as f:
        epoch_str = str(epoch) if epoch is not None else 'N/A'
        loss_str = f"{metrics['loss']:.4f}" if metrics['loss'] is not None else 'N/A'
        acc_str = f"{metrics['accuracy']:.4f}"
        prec_str = f"{metrics['precision']:.4f}"
        rec_str = f"{metrics['recall']:.4f}"
        f1_str = f"{metrics['f1_score']:.4f}"
        roc_auc_str = f"{metrics['auc_roc']:.4f}" if metrics['auc_roc'] is not None else 'N/A'
        prc_auc_str = f"{metrics['auc_prc']:.4f}" if metrics['auc_prc'] is not None else 'N/A'
        
        line = f"{checkpoint_path}\t{dataset_name}\t{epoch_str}\t{loss_str}\t{acc_str}\t{prec_str}\t{rec_str}\t{f1_str}\t{roc_auc_str}\t{prc_auc_str}\n"
        f.write(line)
    
    print(f"  Results appended to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate classifier (matching mammo_dp style)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data_cbis', type=str, default=None, help='Path to CBIS-DDSM test data (in-domain)')
    parser.add_argument('--test_data_bcdr', type=str, default=None, help='Path to BCDR test data (out-of-domain)')
    parser.add_argument('--test_data_eucan', type=str, default=None,
                       help='Path to EUCanImage lesion-classification test data (external validation)')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data (legacy, use test_data_cbis or test_data_bcdr)')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet18', 'resnet50', 'swin_t', 'swin_s', 
                                'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=None,
                       help='Classifier input size. Defaults to checkpoint metadata, then 224.')
    parser.add_argument('--resize_mode', choices=['squash', 'pad'], default=None,
                       help='Defaults to checkpoint metadata, then squash.')
    parser.add_argument('--normalization', choices=['auto', 'imagenet', '0.5'], default='auto',
                       help='Defaults to checkpoint metadata, then ImageNet statistics.')
    parser.add_argument('--allow_preprocessing_override', action='store_true',
                       help='Allow explicit preprocessing values that conflict with checkpoint metadata.')
    parser.add_argument('--save_roc_curves', action='store_true', default=True,
                       help='Save ROC curve plots (default: True)')
    parser.add_argument('--write_results_txt', action='store_true', default=True,
                       help='Write results to text file (matching mammo_dp style, default: True)')
    parser.add_argument('--normalize_05', action='store_true',
                       help='Normalize with mean=0.5, std=0.5 instead of ImageNet stats')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save per-image predictions and metadata as CSV.')
    parser.add_argument('--test_manifest', type=str, default=None,
                       help='Optional manifest for patient, center, view, and laterality metadata.')
    parser.add_argument('--bootstrap_iterations', type=int, default=0,
                       help='Patient-cluster bootstrap iterations; 0 disables bootstrap.')
    parser.add_argument('--bootstrap_seed', type=int, default=2026)
    parser.add_argument('--positive_class', type=str, default='malignant',
                       choices=['malignant', 'benign', 'both'],
                       help='Positive class for AUROC/AUPRC. Use benign to compare with Richard/mammo_dp is_benign reporting; both reports both classes.')
    
    args = parser.parse_args()
    
    # Expand paths
    checkpoint_path = Path(args.checkpoint).expanduser()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, args.model, device)

    checkpoint_image_size = checkpoint.get('image_size') if isinstance(checkpoint, dict) else None
    checkpoint_resize_mode = checkpoint.get('resize_mode') if isinstance(checkpoint, dict) else None
    checkpoint_normalization = checkpoint.get('normalization') if isinstance(checkpoint, dict) else None

    def resolve_preprocessing(name, explicit, checkpoint_value, fallback):
        if (
            explicit is not None
            and checkpoint_value is not None
            and str(explicit) != str(checkpoint_value)
            and not args.allow_preprocessing_override
        ):
            raise ValueError(
                f"Evaluation {name}={explicit} conflicts with checkpoint "
                f"{name}={checkpoint_value}. Pass --allow_preprocessing_override "
                "only for an intentional ablation."
            )
        return explicit if explicit is not None else (
            checkpoint_value if checkpoint_value is not None else fallback
        )

    image_size = int(resolve_preprocessing(
        "image_size",
        args.image_size,
        checkpoint_image_size,
        224,
    ))
    resize_mode = resolve_preprocessing(
        "resize_mode",
        args.resize_mode,
        checkpoint_resize_mode,
        "squash",
    )
    explicit_normalization = (
        "0.5" if args.normalize_05 else (
            None if args.normalization == "auto" else args.normalization
        )
    )
    normalization_name = resolve_preprocessing(
        "normalization",
        explicit_normalization,
        checkpoint_normalization,
        "imagenet",
    )
    
    # Prepare transform (match training normalization)
    if normalization_name == "0.5":
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        print("Using normalization: mean=0.5, std=0.5")
    else:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print("Using normalization: ImageNet stats")

    resize_transform = (
        ResizeContainPad(image_size, fill=0)
        if resize_mode == "pad"
        else transforms.Resize((image_size, image_size))
    )
    print(
        f"Classifier preprocessing: image_size={image_size}, "
        f"resize_mode={resize_mode}, normalization={normalization_name}"
    )
    transform = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        norm,
    ])
    
    # Collect datasets (evaluate separately)
    datasets_to_eval = {}
    
    if args.test_data_cbis:
        cbis_path = Path(args.test_data_cbis).expanduser()
        if cbis_path.exists():
            datasets_to_eval['cbis_ddsm'] = cbis_path
            print(f"Loading CBIS-DDSM dataset from: {cbis_path}")
        else:
            print(f"Warning: CBIS-DDSM path does not exist: {cbis_path}")
    
    if args.test_data_bcdr:
        bcdr_path = Path(args.test_data_bcdr).expanduser()
        if bcdr_path.exists():
            datasets_to_eval['bcdr'] = bcdr_path
            print(f"Loading BCDR dataset from: {bcdr_path}")
        else:
            print(f"Warning: BCDR path does not exist: {bcdr_path}")

    if args.test_data_eucan:
        eucan_path = Path(args.test_data_eucan).expanduser()
        if eucan_path.exists():
            datasets_to_eval['eucanimage'] = eucan_path
            print(f"Loading EUCanImage dataset from: {eucan_path}")
        else:
            print(f"Warning: EUCanImage path does not exist: {eucan_path}")
    
    if args.test_data and not datasets_to_eval:  # Legacy support
        test_path = Path(args.test_data).expanduser()
        if test_path.exists():
            datasets_to_eval['test'] = test_path
            print(f"Loading test dataset from: {test_path}")
        else:
            print(f"Warning: Test path does not exist: {test_path}")
    
    if not datasets_to_eval:
        raise ValueError(
            "Must provide at least one of: --test_data, --test_data_cbis, "
            "--test_data_bcdr, or --test_data_eucan"
        )
    
    # Setup logging (matching mammo_dp style)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Evaluate each dataset separately
    all_results = {
        'checkpoint': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_acc': checkpoint.get('val_acc', 'unknown'),
        'preprocessing': {
            'image_size': image_size,
            'resize_mode': resize_mode,
            'normalization': normalization_name,
        },
        'datasets': {}
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for name, path in datasets_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {name.upper()}: {path}")
        print(f"{'='*60}")
        
        test_dataset = datasets.ImageFolder(str(path), transform=transform)
        print(f"Class to idx: {test_dataset.class_to_idx}")
        if name == 'eucanimage':
            expected_mapping = {'is_benign_false': 0, 'is_benign_true': 1}
            if test_dataset.class_to_idx != expected_mapping:
                raise ValueError(
                    "EUCanImage must use the same class folders and output indices as training: "
                    f"{expected_mapping}. Found: {test_dataset.class_to_idx}. "
                    "Rename/prepare the folders before evaluation; otherwise model logits are assigned "
                    "to the wrong clinical labels."
                )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Evaluating on {len(test_dataset)} samples...")
        positive_classes = ['malignant', 'benign'] if args.positive_class == 'both' else [args.positive_class]
        dataset_results = {}
        malignant_arrays = None

        for positive_class in positive_classes:
            pos_idx = get_class_index(test_dataset.class_to_idx, positive_class)
            print(f"\nUsing positive_class={positive_class} (index={pos_idx}) for AUROC/AUPRC")
            metrics = evaluate_model(model, test_loader, device, positive_class_index=pos_idx, criterion=criterion)
            metrics['test_samples'] = len(test_dataset)
            metrics['dataset_path'] = str(path)
            metrics['positive_class'] = positive_class

            # Extract data for ROC curve (before removing from metrics)
            y_true = metrics.pop('_y_true')
            y_prob = metrics.pop('_y_prob')
            binary_predictions = metrics.pop('_binary_predictions')
            raw_labels = metrics.pop('_raw_labels')
            raw_predictions = metrics.pop('_raw_predictions')
            if positive_class == "malignant":
                malignant_arrays = {
                    "binary_labels": y_true,
                    "probabilities": y_prob,
                    "binary_predictions": binary_predictions,
                    "raw_labels": raw_labels,
                    "raw_predictions": raw_predictions,
                }

            dataset_results[positive_class] = metrics

            # Print results for this dataset (matching mammo_dp style)
            epoch = checkpoint.get('epoch', None)
            print(f"\n{name.upper()} Results ({positive_class} positive):")
            print(f"  Loss: {metrics['loss']:.4f}" if metrics['loss'] is not None else "  Loss: N/A")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}" if metrics['auc_roc'] is not None else "  AUC-ROC: N/A")
            print(f"  AUC-PRC: {metrics['auc_prc']:.4f}" if metrics['auc_prc'] is not None else "  AUC-PRC: N/A")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")

            # Log results (matching mammo_dp style)
            logging.info(f"{checkpoint_path} {name} positive={positive_class}")
            loss_str = f"{metrics['loss']:.4f}" if metrics['loss'] is not None else 'N/A'
            roc_auc_str = f"{metrics['auc_roc']:.4f}" if metrics['auc_roc'] is not None else 'N/A'
            prc_auc_str = f"{metrics['auc_prc']:.4f}" if metrics['auc_prc'] is not None else 'N/A'
            logging.info(f"{epoch}\t{loss_str}\t{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1_score']:.4f}\t{roc_auc_str}\t{prc_auc_str}")

            # Write to results text file (matching mammo_dp style)
            if args.write_results_txt:
                txt_name = name if args.positive_class != 'both' else f"{name}_{positive_class}_positive"
                write_results_txt(args.output_file, str(checkpoint_path), txt_name, metrics, epoch)

            # Generate ROC curve (matching mammo_dp style)
            if args.save_roc_curves:
                roc_name = name if args.positive_class != 'both' else f"{name}_{positive_class}_positive"
                output_ROC_curve(y_true, y_prob, args.output_file, description=roc_name)

        if malignant_arrays is not None and (
            args.save_predictions or args.bootstrap_iterations > 0
        ):
            manifest_metadata, manifest_path = load_manifest_metadata(
                path,
                args.test_manifest,
            )
            prediction_rows = build_prediction_rows(
                test_dataset,
                manifest_metadata,
                malignant_arrays,
            )
            view_summaries = stratified_summaries(prediction_rows, "view")
            center_summaries = stratified_summaries(prediction_rows, "center")
            analysis = {
                "manifest": str(manifest_path) if manifest_path else None,
                "stratified_by_view": view_summaries,
                "stratified_by_center": center_summaries,
                "patient_cluster_bootstrap": patient_cluster_bootstrap(
                    prediction_rows,
                    args.bootstrap_iterations,
                    args.bootstrap_seed,
                ),
            }
            if args.save_predictions:
                output_stem = Path(args.output_file).stem
                prediction_path = (
                    Path(args.output_file).parent
                    / f"predictions_{name}_{output_stem}.csv"
                )
                write_prediction_csv(prediction_path, prediction_rows)
                analysis["predictions_csv"] = str(prediction_path)
                print(f"  Per-image predictions saved to: {prediction_path}")
                stratified_rows = []
                for group_field, summaries in (
                    ("view", view_summaries),
                    ("center", center_summaries),
                ):
                    for group_value, summary in summaries.items():
                        stratified_rows.append(
                            {
                                "group_field": group_field,
                                "group_value": group_value,
                                **summary,
                            }
                        )
                if stratified_rows:
                    stratified_path = (
                        Path(args.output_file).parent
                        / f"stratified_metrics_{name}_{output_stem}.csv"
                    )
                    write_prediction_csv(stratified_path, stratified_rows)
                    analysis["stratified_metrics_csv"] = str(stratified_path)
                    print(f"  Stratified metrics saved to: {stratified_path}")
            dataset_results["analysis"] = analysis

        if args.positive_class == 'both':
            all_results['datasets'][name] = dataset_results
        else:
            all_results['datasets'][name] = dataset_results[args.positive_class]
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
