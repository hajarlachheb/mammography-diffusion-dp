#!/usr/bin/env python3
"""
Main Classifier Training Script
Supports standard training and DP-SGD training for mammography classification
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import logging
from datetime import datetime
from torchvision import datasets, transforms, models
from PIL import Image, ImageOps
import signal
import numpy as np
from sklearn.metrics import average_precision_score
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    Swin_T_Weights,
    Swin_S_Weights,
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# DINOv3 support via timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. DINOv3 models will not be available.")

# Differential Privacy
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("Warning: Opacus not installed. DP training will not be available.")

# Import custom modules (to be implemented)
# from models.classifier import MammographyClassifier


def make_swin_opacus_compatible(model):
    """Convert registered buffers to plain attributes for Opacus compatibility.

    Opacus rejects trainable modules that have both trainable parameters
    AND registered buffers. SwinT's ShiftedWindowAttention stores precomputed
    attention masks and relative position indices as buffers alongside its
    trainable qkv/proj weights.

    We only strip buffers from modules that actually have trainable parameters,
    leaving non-trainable modules (e.g. BatchNorm, LayerNorm) untouched so
    their buffers (running_mean, running_var) remain functional.
    """
    converted = 0
    for name, module in model.named_modules():
        has_trainable = any(p.requires_grad for p in module.parameters(recurse=False))
        if not has_trainable:
            continue
        buf_names = list(module._buffers.keys())
        for buf_name in buf_names:
            buf = module._buffers.pop(buf_name)
            setattr(module, buf_name, buf)
            converted += 1
    if converted > 0:
        print(f"  Converted {converted} buffers from trainable modules for Opacus compatibility", flush=True)
    return model
# from utils.data_loader import MammographyDataset
# from evaluation.classification_metrics import compute_metrics


def unwrap_model_for_state_dict(model):
    """Return the plain model to save, unwrapping Opacus/DataParallel wrappers."""
    if hasattr(model, "_module"):
        return model._module
    if hasattr(model, "module"):
        return model.module
    return model


def get_clean_state_dict_from_checkpoint(checkpoint):
    """Get a loadable state_dict and strip common wrapper prefixes."""
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("_module.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def is_epsilon_within_target(target_epsilon, actual_epsilon, lower_fraction=0.75, upper_fraction=1.05):
    """Match mammo_dp checkpoint eligibility: accept only near target epsilon."""
    if target_epsilon is None or actual_epsilon is None:
        return True
    lower_bound = target_epsilon * lower_fraction
    upper_bound = target_epsilon * upper_fraction
    return lower_bound <= actual_epsilon <= upper_bound


def setup_logging(output_dir, experiment_id):
    """Setup logging configuration"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_id}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train mammography classifier')
    
    # Experiment
    parser.add_argument('--experiment_id', type=str, required=True,
                       help='Experiment ID (e.g., E1.1)')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Path to an explicit validation ImageFolder. Use this for patient-level splits.')
    parser.add_argument('--test_data_path', type=str, default=None,
                       help='Path to test data (optional, legacy)')
    parser.add_argument('--test_data_cbis', type=str, default=None,
                       help='Path to CBIS-DDSM test data (in-domain)')
    parser.add_argument('--test_data_bcdr', type=str, default=None,
                       help='Path to BCDR test data (out-of-domain)')
    parser.add_argument('--subsample_ratio', type=float, default=1.0,
                       help='Ratio of data to use (for CPU testing, e.g., 0.5 for 50%%)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'swin_t', 'swin_s', 
                                'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Square classifier input size')
    parser.add_argument('--resize_mode', choices=['squash', 'pad'], default='squash',
                       help='squash resizes directly; pad preserves aspect ratio on a black canvas')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--backbone_learning_rate', type=float, default=None,
                       help='Optional learning rate for trainable non-head parameters')
    parser.add_argument('--head_learning_rate', type=float, default=None,
                       help='Optional learning rate for the newly initialized classifier head')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer type (adam or adamw)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for final layer')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing for CrossEntropyLoss (0.0 = off, 0.1 = Richard default)')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--early_stopping_metric', choices=['loss', 'auprc'], default='loss',
                       help='Metric used for patience tracking; use auprc when selecting by AUPRC')
    parser.add_argument('--no_early_stopping', action='store_true',
                       help='Disable early stopping, train for all epochs')
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                       help='Use learning rate scheduler')
    parser.add_argument('--simple_augmentation', action='store_true',
                       help='Use simple augmentation (HFlip+VFlip only, matching Richard)')
    parser.add_argument('--horizontal_flip_only', action='store_true',
                       help='Use resize + horizontal flip only. Recommended for full mammograms.')
    parser.add_argument('--normalize_05', action='store_true',
                       help='Normalize with mean=0.5, std=0.5 instead of ImageNet stats')
    parser.add_argument('--select_by_auprc', action='store_true',
                       help='Select best model by validation AUPRC instead of accuracy')
    parser.add_argument('--richard_finetuning', action='store_true',
                       help='Apply mammo_dp/Richard finetuning defaults: AdamW, lr=1e-5, wd=1e-8, label_smoothing=0.1, '
                            '0.5 normalization, H/V flips, AUPRC selection, no early stopping, no scheduler.')
    parser.add_argument('--validation_positive_class', type=str, default='malignant',
                       choices=['malignant', 'benign'],
                       help='Positive class for validation AUROC/AUPRC selection. Richard/mammo_dp uses benign.')
    
    # Backbone freezing
    parser.add_argument('--head_type', choices=['linear', 'ref_mlp'], default='linear',
                       help="Swin classifier head. 'linear' (default, unchanged): Dropout(--dropout) + Linear(768, 2). "
                            "'ref_mlp': Richard Osuala's mammo_dp frozen-Swin head, Linear(768, 384) + LeakyReLU(0.2) + "
                            "Linear(384, 2), no dropout (gan_compare/constants.py freeze_layers)")
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone, train only the last N layers (see --unfreeze_last_n)')
    parser.add_argument('--unfreeze_last_n', type=int, default=1,
                       help='How many layer groups to unfreeze when --freeze_backbone is set '
                            '(1 = head only, 2 = head + norm + last feature stage)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Class imbalance handling
    parser.add_argument('--use_weighted_sampler', action='store_true',
                       help='Use WeightedRandomSampler for class imbalance')
    
    # Differential Privacy
    parser.add_argument('--enable_dp', action='store_true',
                       help='Enable differential privacy')
    parser.add_argument('--no_dp', action='store_true',
                       help='Disable differential privacy (explicit)')
    parser.add_argument('--epsilon', type=float, default=10.0,
                       help='Target epsilon for DP')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='Target delta for DP')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--noise_multiplier', type=float, default=None,
                       help='Explicit noise multiplier for DP-SGD. If not set, auto-calibrated from --epsilon via make_private_with_epsilon')
    parser.add_argument('--max_physical_batch_size', type=int, default=32,
                       help='Max physical batch size for Opacus BatchMemoryManager (virtual batching). '
                            'Logical batch_size is preserved for privacy accounting.')
    parser.add_argument('--dp_accountant', type=str, default='rdp',
                       choices=['rdp', 'gdp', 'prv'],
                       help='Opacus accountant. mammo_dp/Richard used rdp; Opacus default is prv.')
    parser.add_argument('--dp_require_epsilon_near_target_for_selection', action='store_true',
                       help='For DP runs, only allow best-checkpoint selection once actual epsilon is near target.')
    parser.add_argument('--dp_selection_min_epsilon_fraction', type=float, default=0.75,
                       help='Lower bound fraction of target epsilon for DP checkpoint selection eligibility.')
    parser.add_argument('--dp_selection_max_epsilon_fraction', type=float, default=1.05,
                       help='Upper bound fraction of target epsilon for DP checkpoint selection eligibility.')
    
    # Fine-tuning from pretrained checkpoint
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                       help='Path to pretrained model checkpoint for fine-tuning (e.g., from synthetic pretraining)')
    parser.add_argument('--finetune_head_strategy', type=str, default='keep',
                       choices=['keep', 'reinit'],
                       help='How to handle the classifier head after loading --pretrained_checkpoint. '
                            'keep = Option 2, keep synthetic-pretrained CLF head and fine-tune it; '
                            'reinit = Option 1, replace the CLF head before real fine-tuning.')
    
    # Config files
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Base configuration file')
    parser.add_argument('--dp_config', type=str, default='config/dp_config.yaml',
                       help='DP configuration file')
    
    # Testing
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode (1 epoch, small dataset)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--cpu_optimized', action='store_true',
                       help='Use CPU-optimized settings (smaller batch, reduced workers)')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, use_dp=False, privacy_engine=None, max_physical_batch_size=32, dp_delta=1e-4):
    """Train for one epoch, using BatchMemoryManager for DP virtual batching."""
    print("  Setting model to train mode...", flush=True)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0

    if use_dp and max_physical_batch_size:
        print(f"  Using BatchMemoryManager (physical_batch={max_physical_batch_size})...", flush=True)
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for batch_idx, (images, labels) in enumerate(memory_safe_data_loader):
                if batch_idx == 0:
                    print(f"  Processing first batch (batch {batch_idx+1})...", flush=True)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    if privacy_engine:
                        epsilon = privacy_engine.get_epsilon(delta=dp_delta)
                        logging.info(f'Batch [{batch_idx+1}] '
                                   f'Loss: {loss.item():.4f} '
                                   f'Acc: {100.*correct/total:.2f}% '
                                   f'ε: {epsilon:.2f}')
                    else:
                        logging.info(f'Batch [{batch_idx+1}] '
                                   f'Loss: {loss.item():.4f} '
                                   f'Acc: {100.*correct/total:.2f}%')
    else:
        print(f"  Starting to iterate over {len(dataloader)} batches...", flush=True)
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx == 0:
                print(f"  Processing first batch (batch {batch_idx+1})...", flush=True)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                logging.info(f'Batch [{batch_idx+1}/{len(dataloader)}] '
                           f'Loss: {loss.item():.4f} '
                           f'Acc: {100.*correct/total:.2f}%')

    num_batches = batch_idx + 1 if total > 0 else 1
    epoch_loss = running_loss / num_batches
    epoch_acc = 100. * correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model, returning loss, accuracy, and AUPRC."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            try:
                class_to_idx = dataloader.dataset.class_to_idx
            except Exception:
                class_to_idx = {}
            pos_idx = infer_positive_class_index(class_to_idx)
            labels_np = labels.cpu().numpy()
            probs = torch.softmax(outputs, dim=1)[:, pos_idx].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend((labels_np == pos_idx).astype(int))
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    try:
        val_auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        val_auprc = 0.0
    
    return val_loss, val_acc, val_auprc


def infer_positive_class_index(class_to_idx: dict, positive_class: str = "malignant") -> int:
    """
    Infer which class index should be treated as the *positive* class for AUPRC selection.

    Convention in this repo's datasets:
      - is_benign_false == malignant
      - is_benign_true  == benign
    """
    if not class_to_idx:
        return 1

    if positive_class == "benign":
        preferred_keys = ("is_benign_true", "benign")
    else:
        preferred_keys = ("is_benign_false", "malignant", "cancer", "positive", "pos")

    for key in preferred_keys:
        if key in class_to_idx:
            return int(class_to_idx[key])

    return 1


def validate_with_pos_index(model, dataloader, criterion, device, positive_class_index: int):
    """Validate and compute AUPRC w.r.t. a chosen positive class index."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            labels_np = labels.cpu().numpy()
            probs = torch.softmax(outputs, dim=1)[:, positive_class_index].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend((labels_np == positive_class_index).astype(int))

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    try:
        val_auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        val_auprc = 0.0

    return val_loss, val_acc, val_auprc


def get_targets_from_dataset(ds):
    """
    Return a list of integer class targets for datasets used in this script.

    Supports:
      - torchvision.datasets.ImageFolder (has .targets)
      - torch.utils.data.Subset wrapping an ImageFolder (has .dataset + .indices)
    """
    # ImageFolder / similar
    if hasattr(ds, "targets"):
        return list(ds.targets)

    # Subset(ImageFolder)
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        idxs = ds.indices
        if hasattr(base, "targets"):
            base_targets = base.targets
            return [int(base_targets[i]) for i in idxs]

    raise AttributeError("Unsupported dataset type for target extraction (missing .targets or Subset(dataset.targets)).")


def reinitialize_classifier_head(model, model_name: str, dropout: float):
    """Replace the classifier head while preserving the feature extractor."""
    if model_name in ['resnet18', 'resnet50']:
        in_features = model.fc[-1].in_features if isinstance(model.fc, nn.Sequential) else model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
    elif model_name == 'efficientnet_b0':
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
    elif model_name in ['swin_t', 'swin_s']:
        in_features = model.head[-1].in_features if isinstance(model.head, nn.Sequential) else model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
    elif model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        in_features = model.head[-1].in_features if isinstance(model.head, nn.Sequential) else model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
    else:
        raise ValueError(f"Unsupported model for head reinitialization: {model_name}")
    return model


def get_classifier_head(model, model_name: str):
    if model_name in ['resnet18', 'resnet50']:
        return model.fc
    if model_name == 'efficientnet_b0':
        return model.classifier
    if model_name in ['swin_t', 'swin_s', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        return model.head
    raise ValueError(f"Unsupported model for classifier-head lookup: {model_name}")


def build_optimizer_param_groups(model, model_name, backbone_lr, head_lr):
    """Create disjoint backbone/head parameter groups for fine-tuning."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head_ids = {
        id(parameter)
        for parameter in get_classifier_head(model, model_name).parameters()
        if parameter.requires_grad
    }
    head_parameters = [parameter for parameter in trainable if id(parameter) in head_ids]
    backbone_parameters = [parameter for parameter in trainable if id(parameter) not in head_ids]
    groups = []
    if backbone_parameters:
        groups.append(
            {
                'params': backbone_parameters,
                'lr': backbone_lr,
                'group_name': 'backbone',
            }
        )
    if head_parameters:
        groups.append(
            {
                'params': head_parameters,
                'lr': head_lr,
                'group_name': 'head',
            }
        )
    if not groups:
        raise RuntimeError("No trainable optimizer parameters were found")
    return groups


def apply_richard_finetuning_defaults(args):
    """Apply the mammo_dp classifier fine-tuning recipe as one coherent mode."""
    if not args.richard_finetuning:
        return

    args.simple_augmentation = True
    args.normalize_05 = True
    args.select_by_auprc = True
    args.no_early_stopping = True
    args.use_scheduler = False
    args.optimizer = "adamw"
    args.learning_rate = 1e-5
    args.weight_decay = 1e-8
    args.label_smoothing = 0.1
    args.validation_positive_class = "benign"
    args.dp_accountant = "rdp"
    args.dp_require_epsilon_near_target_for_selection = True

    # Richard's config uses these defaults for Swin fine-tuning. Preserve explicit
    # non-default CLI values so small smoke tests can still override them.
    if args.epochs == 50:
        args.epochs = 300
    if args.batch_size == 32:
        args.batch_size = 128


def main():
    global args
    print("Starting training script...", flush=True)
    args = parse_args()
    apply_richard_finetuning_defaults(args)
    if args.head_type != 'linear' and args.model != 'swin_t':
        raise ValueError(f"--head_type {args.head_type} is only implemented for swin_t")
    if args.head_type != 'linear' and args.pretrained_checkpoint and args.finetune_head_strategy == 'reinit':
        raise ValueError("--finetune_head_strategy reinit rebuilds a linear head; not supported with --head_type ref_mlp")
    print(f"Arguments parsed. Experiment ID: {args.experiment_id}", flush=True)
    if args.richard_finetuning:
        print("Richard/mammo_dp finetuning mode enabled", flush=True)
        print("  AdamW lr=1e-5 wd=1e-8, label_smoothing=0.1, norm=0.5, H/V flips, select=benign AUPRC, no early stopping, no scheduler", flush=True)
    
    # Seed for reproducibility
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to {args.seed}", flush=True)
    
    # Expand paths with tilde (~)
    args.output_dir = str(Path(args.output_dir).expanduser())
    args.data_path = str(Path(args.data_path).expanduser())
    if args.val_data_path:
        args.val_data_path = str(Path(args.val_data_path).expanduser())
    if args.test_data_path:
        args.test_data_path = str(Path(args.test_data_path).expanduser())
    if args.test_data_cbis:
        args.test_data_cbis = str(Path(args.test_data_cbis).expanduser())
    if args.test_data_bcdr:
        args.test_data_bcdr = str(Path(args.test_data_bcdr).expanduser())
    
    print(f"Expanded data path: {args.data_path}", flush=True)
    if args.val_data_path:
        print(f"Expanded validation data path: {args.val_data_path}", flush=True)
    print(f"Expanded output dir: {args.output_dir}", flush=True)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}", flush=True)
    
    # CPU optimizations
    if args.cpu_optimized or args.device == 'cpu':
        logger_temp = logging.getLogger(__name__)
        logger_temp.warning("CPU mode detected - applying optimizations")
        if args.batch_size > 8:
            args.batch_size = 8
            logger_temp.info(f"Reduced batch size to {args.batch_size} for CPU")
        if args.num_workers > 2:
            args.num_workers = 2
            logger_temp.info(f"Reduced num_workers to {args.num_workers} for CPU")
    
    # Setup
    print("Setting up logging...", flush=True)
    logger = setup_logging(args.output_dir, args.experiment_id)
    print("Logging setup complete.", flush=True)
    logger.info(f"Starting experiment {args.experiment_id}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    if device.type == 'cpu' and args.device == 'cuda':
        logger.warning("CUDA requested but not available. Falling back to CPU")
    logger.info(f"Using device: {device}")
    
    # Load configurations (optional - config file not currently used)
    base_config = None
    if args.config and Path(args.config).exists():
        print(f"Loading config from: {args.config}")
        try:
            base_config = load_config(args.config)
            print("Base config loaded successfully")
        except Exception as e:
            print(f"WARNING: Failed to load base config: {e} (continuing without config)", file=sys.stderr)
            logger.warning(f"Config file not found, continuing without it: {e}")
    else:
        print(f"Config file not found at {args.config}, continuing without config file")
        logger.info("Running without config file (using command-line arguments only)")
    
    dp_config = None
    if args.enable_dp and not args.no_dp:
        if args.noise_multiplier is not None:
            print(f"DP enabled with explicit noise_multiplier={args.noise_multiplier}")
            logger.info("Differential Privacy ENABLED (explicit noise_multiplier)")
        else:
            print(f"DP enabled with auto-calibration: target ε={args.epsilon}, δ={args.delta}, epochs={args.epochs}")
            logger.info("Differential Privacy ENABLED (auto-calibrated via make_private_with_epsilon)")
        if Path(args.dp_config).exists():
            try:
                dp_config = load_config(args.dp_config)
                print(f"Loaded DP config from: {args.dp_config}")
            except Exception as e:
                print(f"WARNING: Failed to load DP config: {e} (using CLI args only)")
                logger.warning(f"Failed to load DP config: {e}")
        else:
            print(f"No DP config file at {args.dp_config}, using CLI args for DP parameters")
    else:
        logger.info("Differential Privacy DISABLED")
    
    # Normalization: match Richard's (0.5, 0.5) or use ImageNet stats
    if args.normalize_05:
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        print("Using normalization: mean=0.5, std=0.5 (matching Richard)", flush=True)
    else:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print("Using normalization: ImageNet stats", flush=True)

    if args.image_size <= 0:
        raise ValueError("--image_size must be positive")
    resize_transform = (
        ResizeContainPad(args.image_size, fill=0)
        if args.resize_mode == 'pad'
        else transforms.Resize((args.image_size, args.image_size))
    )
    print(
        f"Classifier preprocessing: image_size={args.image_size}, "
        f"resize_mode={args.resize_mode}",
        flush=True,
    )

    if args.horizontal_flip_only:
        train_transform = transforms.Compose([
            resize_transform,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
        print("Using full-mammogram augmentation (horizontal flip only)", flush=True)
    elif args.simple_augmentation:
        train_transform = transforms.Compose([
            resize_transform,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            norm,
        ])
        print("Using simple augmentation (HFlip + VFlip, matching Richard)", flush=True)
    else:
        train_transform = transforms.Compose([
            resize_transform,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.1),
        ])
        print("Using full augmentation (HFlip + Rotation + ColorJitter + Affine + Erasing)", flush=True)

    val_transform = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        norm,
    ])

    # Datasets / loaders (ImageFolder expects class folders, e.g., is_benign_false/is_benign_true)
    print(f"Loading training dataset from: {args.data_path}", flush=True)
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Training data path does not exist: {args.data_path}")
    train_dataset = datasets.ImageFolder(args.data_path, transform=train_transform)
    print(f"Training dataset loaded: {len(train_dataset)} samples", flush=True)
    print(f"Train class_to_idx: {train_dataset.class_to_idx}", flush=True)
    
    # Never use a test set for validation. For grouped data such as EUCanImage,
    # provide a patient-disjoint validation ImageFolder with --val_data_path.
    if args.val_data_path:
        if not Path(args.val_data_path).exists():
            raise FileNotFoundError(
                f"Validation data path does not exist: {args.val_data_path}"
            )
        val_dataset = datasets.ImageFolder(args.val_data_path, transform=val_transform)
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError(
                "Training and validation class mappings differ: "
                f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
            )
        print(
            "Using explicit validation dataset: "
            f"train={len(train_dataset)}, val={len(val_dataset)}",
            flush=True,
        )
    else:
        val_split = 0.2
        rng = np.random.RandomState(args.seed)

        # Build a second ImageFolder with val_transform so we can subset with the same indices.
        full_for_val = datasets.ImageFolder(args.data_path, transform=val_transform)
        targets = np.array(train_dataset.targets)
        idx_all = np.arange(len(targets))
        val_indices = []
        train_indices = []
        for cls in np.unique(targets):
            cls_idx = idx_all[targets == cls]
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(val_split * len(cls_idx))))
            val_indices.extend(cls_idx[:n_val].tolist())
            train_indices.extend(cls_idx[n_val:].tolist())

        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(full_for_val, val_indices)
        print(
            f"Train/val split from training data: train={len(train_dataset)}, "
            f"val={len(val_dataset)} (seed={args.seed})",
            flush=True,
        )

    # Setup weighted sampler for class imbalance (optional)
    train_sampler = None
    shuffle_train = True
    if args.use_weighted_sampler:
        targets = get_targets_from_dataset(train_dataset)
        class_counts = torch.bincount(torch.tensor(targets))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[targets]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle_train = False  # Cannot use shuffle with sampler
        print(f"Using WeightedRandomSampler (class counts: {class_counts.tolist()})", flush=True)

    print("Creating data loaders...", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    print(f"Training loader created with {len(train_loader)} batches", flush=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Validation loader created with {len(val_loader)} batches", flush=True)

    # Model factory with dropout to reduce overfitting
    print(f"Creating model: {args.model} (pretrained={args.pretrained})")
    if args.model == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = models.resnet18(weights=weights)
        # Add dropout to final layer
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(model.fc.in_features, 2)
        )
        print(f"Added dropout ({args.dropout}) to final layer", flush=True)
    elif args.model == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2 if args.pretrained else None
        print("Loading ResNet50 pretrained weights...", flush=True)
        model = models.resnet50(weights=weights)
        print("ResNet50 loaded, modifying final layer...", flush=True)
        # Add dropout to final layer
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(model.fc.in_features, 2)
        )
        print(f"Added dropout ({args.dropout}) to final layer", flush=True)
        print("Model architecture ready", flush=True)
    elif args.model == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # Add dropout before final classifier (EfficientNet classifier is already Sequential)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features, 2)
        )
        print(f"Added dropout ({args.dropout}) to classifier", flush=True)
    elif args.model == 'swin_t':
        weights = Swin_T_Weights.IMAGENET1K_V1 if args.pretrained else None
        print("Loading Swin-T pretrained weights...", flush=True)
        model = models.swin_t(weights=weights)
        print("Swin-T loaded, modifying head...", flush=True)
        # Swin uses 'head' instead of 'fc'
        in_features = model.head.in_features
        if args.head_type == 'ref_mlp':
            model.head = nn.Sequential(
                nn.Linear(in_features, 384),
                nn.LeakyReLU(0.2),
                nn.Linear(384, 2)
            )
            print("Using reference mammo_dp head: Linear(768, 384) + LeakyReLU(0.2) + Linear(384, 2), no dropout "
                  "(LeakyReLU not in-place: Opacus backward hooks forbid in-place ops on their views)", flush=True)
        else:
            model.head = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(in_features, 2)
            )
            print(f"Added dropout ({args.dropout}) to head", flush=True)
        print("Swin-T architecture ready", flush=True)
    elif args.model == 'swin_s':
        weights = Swin_S_Weights.IMAGENET1K_V1 if args.pretrained else None
        print("Loading Swin-S pretrained weights...", flush=True)
        model = models.swin_s(weights=weights)
        print("Swin-S loaded, modifying head...", flush=True)
        # Swin uses 'head' instead of 'fc'
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features, 2)
        )
        print(f"Added dropout ({args.dropout}) to head", flush=True)
        print("Swin-S architecture ready", flush=True)
    elif args.model in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for DINOv3 models. Install with: pip install timm")
        print(f"Loading DINOv3 ({args.model}) pretrained weights...", flush=True)
        # DINOv3 models from timm are pretrained by default
        pretrained_flag = args.pretrained
        model = timm.create_model(args.model, pretrained=pretrained_flag, num_classes=0)  # num_classes=0 removes head
        print(f"DINOv3 ({args.model}) loaded, adding classification head...", flush=True)
        # Get embedding dimension from model
        embedding_dim = model.num_features if hasattr(model, 'num_features') else model.embed_dim
        # Add classification head with dropout
        model.head = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(embedding_dim, 2)
        )
        print(f"Added dropout ({args.dropout}) and classification head (embedding_dim={embedding_dim})", flush=True)
        print(f"DINOv3 ({args.model}) architecture ready", flush=True)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(f"Moving model to device: {device}", flush=True)
    model = model.to(device)
    print("Model moved to device successfully", flush=True)
    
    # =============================
    # Load pretrained checkpoint for fine-tuning (if provided)
    # =============================
    if args.pretrained_checkpoint:
        pretrained_path = Path(args.pretrained_checkpoint).expanduser()
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        print(f"Loading pretrained checkpoint from: {pretrained_path}", flush=True)
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        state_dict = get_clean_state_dict_from_checkpoint(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"Checkpoint load warning: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}",
                flush=True,
            )
            logger.warning(f"Checkpoint load warning: missing={missing[:10]}, unexpected={unexpected[:10]}")
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f"Loaded model_state_dict (trained for {checkpoint.get('epoch', 'unknown')} epochs, val_acc={checkpoint.get('val_acc', 'unknown')})", flush=True)
        if args.finetune_head_strategy == 'reinit':
            model = reinitialize_classifier_head(model, args.model, args.dropout).to(device)
            print("Fine-tune head strategy: reinit (Option 1) - classifier head replaced after loading checkpoint.", flush=True)
            logger.info("Fine-tune head strategy: reinit (Option 1)")
        else:
            print("Fine-tune head strategy: keep (Option 2) - synthetic-pretrained classifier head retained for fine-tuning.", flush=True)
            logger.info("Fine-tune head strategy: keep (Option 2)")
        print("Pretrained checkpoint loaded successfully! Ready for fine-tuning.", flush=True)
        logger.info(f"Loaded pretrained checkpoint from {pretrained_path}")
    
    # =============================
    # Backbone freezing (optional)
    # --unfreeze_last_n controls how many layer groups stay trainable:
    #   1 = head only (original last-layer mode)
    #   2 = head + norm + last feature stage
    # =============================
    if args.freeze_backbone:
        n = args.unfreeze_last_n
        for p in model.parameters():
            p.requires_grad = False

        if args.model in ['resnet18', 'resnet50']:
            for p in model.fc.parameters():
                p.requires_grad = True
            if n >= 2:
                for p in model.layer4.parameters():
                    p.requires_grad = True

        elif args.model == 'efficientnet_b0':
            for p in model.classifier.parameters():
                p.requires_grad = True
            if n >= 2:
                for p in model.features[-1].parameters():
                    p.requires_grad = True

        elif args.model in ['swin_t', 'swin_s']:
            for p in model.head.parameters():
                p.requires_grad = True
            if n >= 2:
                for p in model.norm.parameters():
                    p.requires_grad = True
                for p in model.features[-1].parameters():
                    p.requires_grad = True

        elif args.model in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
            for p in model.head.parameters():
                p.requires_grad = True
            if n >= 2:
                for p in model.norm.parameters():
                    p.requires_grad = True
                if hasattr(model, 'blocks'):
                    for p in model.blocks[-1].parameters():
                        p.requires_grad = True
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_n = sum(p.numel() for p in model.parameters())
        print(f"Backbone FROZEN; unfreezing last {n} layer group(s) "
              f"({trainable_n:,} / {total_n:,} params trainable)", flush=True)
    else:
        print("Full model training; all layers will be trained", flush=True)
    
    # Compute class weights from training set to mitigate imbalance
    train_targets = get_targets_from_dataset(train_dataset)
    class_counts = torch.tensor(train_targets).bincount().float()
    class_weights = (class_counts.sum() / class_counts).to(device)
    training_class_to_idx = getattr(train_dataset, "class_to_idx", None)
    if training_class_to_idx is None and hasattr(train_dataset, "dataset"):
        training_class_to_idx = getattr(train_dataset.dataset, "class_to_idx", {})
    training_class_to_idx = dict(training_class_to_idx or {})
    print(f"Class counts: {class_counts.tolist()}, class weights: {class_weights.tolist()}", flush=True)

    # Setup loss (disable class weights for DP to avoid conflict with DP noise)
    print("Setting up loss function and optimizer...", flush=True)
    ls = args.label_smoothing
    if args.enable_dp and not args.no_dp:
        print(f"DP enabled: using unweighted CrossEntropyLoss (label_smoothing={ls})", flush=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    elif args.richard_finetuning:
        print(f"Richard finetuning: using unweighted CrossEntropyLoss (label_smoothing={ls})", flush=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    elif args.use_weighted_sampler:
        print(
            "Weighted sampler enabled: using unweighted CrossEntropyLoss "
            f"(label_smoothing={ls}) to avoid correcting class imbalance twice",
            flush=True,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    else:
        print(f"Non-DP: using class-weighted CrossEntropyLoss (label_smoothing={ls})", flush=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)

    # Collect trainable params and optionally use a larger learning rate for the
    # newly initialized head than for the ImageNet-pretrained backbone.
    backbone_lr = (
        args.backbone_learning_rate
        if args.backbone_learning_rate is not None
        else args.learning_rate
    )
    head_lr = (
        args.head_learning_rate
        if args.head_learning_rate is not None
        else args.learning_rate
    )
    optimizer_param_groups = build_optimizer_param_groups(
        model,
        args.model,
        backbone_lr,
        head_lr,
    )
    trainable_params = [
        parameter
        for group in optimizer_param_groups
        for parameter in group['params']
    ]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(
        f"Learning rates: backbone={backbone_lr:g}, head={head_lr:g}",
        flush=True,
    )

    # Setup Differential Privacy / optimizer
    privacy_engine = None
    optimizer = None
    if args.enable_dp and not args.no_dp:
        if not OPACUS_AVAILABLE:
            logger.error("Opacus not available. Cannot enable DP training.")
            sys.exit(1)
        print("Making model Opacus-compatible (removing buffers, fixing layers)...", flush=True)
        model = make_swin_opacus_compatible(model)
        model = ModuleValidator.fix(model)
        optimizer_param_groups = build_optimizer_param_groups(
            model,
            args.model,
            backbone_lr,
            head_lr,
        )
        if args.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_param_groups, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(optimizer_param_groups, weight_decay=args.weight_decay)

        # Match the experimental runs; production use requires secure_mode=True and retraining.
        privacy_engine = PrivacyEngine(accountant=args.dp_accountant, secure_mode=False)

        noise_mul = args.noise_multiplier
        if noise_mul is None and dp_config and 'differential_privacy' in dp_config:
            noise_mul = dp_config['differential_privacy'].get('noise_multiplier')

        if noise_mul is not None:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"DP attached via make_private (noise_multiplier={noise_mul}, max_grad_norm={args.max_grad_norm})", flush=True)
        else:
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=args.epochs,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"DP attached via make_private_with_epsilon (target ε={args.epsilon}, δ={args.delta}, epochs={args.epochs})", flush=True)

        logger.info(f"Privacy engine attached: target ε={args.epsilon}, δ={args.delta}")
        print(f"DP accountant: {args.dp_accountant}", flush=True)
        trainable_after_dp = sum(
            p.numel() for p in unwrap_model_for_state_dict(model).parameters() if p.requires_grad
        )
        print(f"Trainable parameters after Opacus wrapping: {trainable_after_dp:,}", flush=True)
    else:
        if args.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_param_groups, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(optimizer_param_groups, weight_decay=args.weight_decay)

    print(
        f"Optimizer created successfully ({args.optimizer.upper()}, "
        f"weight_decay={args.weight_decay})",
        flush=True,
    )

    # Setup learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        print("Learning rate scheduler enabled (ReduceLROnPlateau)", flush=True)
    
    # Training loop with early stopping
    print("Initializing training loop...", flush=True)
    best_val_acc = 0.0
    best_val_auprc = 0.0
    best_val_loss = float('inf')
    best_selection_loss = float('inf')
    best_early_stopping_auprc = float('-inf')
    saved_best_checkpoint = False
    patience_counter = 0
    if args.select_by_auprc:
        print("Model selection: best validation AUPRC (matching Richard)", flush=True)
    else:
        print("Model selection: best validation accuracy", flush=True)
    if args.no_early_stopping:
        print("Early stopping: DISABLED (training all epochs)", flush=True)
    else:
        print(
            f"Early stopping: metric={args.early_stopping_metric}, "
            f"patience={args.early_stopping_patience}",
            flush=True,
        )
    results = {
        'experiment_id': args.experiment_id,
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epsilon': [] if args.enable_dp else None,
        'early_stopped': False,
        'stopped_at_epoch': None
    }
    print(f"Starting training for {args.epochs} epochs...", flush=True)
    print(f"Early stopping patience: {args.early_stopping_patience} epochs", flush=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]", flush=True)
        logger.info(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Train
        print(f"Starting training epoch {epoch+1}...", flush=True)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_dp=args.enable_dp, privacy_engine=privacy_engine,
            max_physical_batch_size=args.max_physical_batch_size,
            dp_delta=args.delta
        )
        print(f"Training epoch {epoch+1} completed. Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%", flush=True)
        
        # Validate
        print(f"Starting validation epoch {epoch+1}...", flush=True)
        # Infer the positive class from either ImageFolder or Subset(ImageFolder).
        class_to_idx = getattr(val_dataset, "class_to_idx", None)
        if class_to_idx is None and hasattr(val_dataset, "dataset"):
            class_to_idx = getattr(val_dataset.dataset, "class_to_idx", {})
        pos_idx = infer_positive_class_index(class_to_idx, args.validation_positive_class)
        val_loss, val_acc, val_auprc = validate_with_pos_index(model, val_loader, criterion, device, positive_class_index=pos_idx)
        print(f"Validation epoch {epoch+1} completed. Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, {args.validation_positive_class} AUPRC: {val_auprc:.4f}", flush=True)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUPRC: {val_auprc:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Track results
        results['epochs'].append(epoch + 1)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results.setdefault('val_auprc', []).append(val_auprc)
        
        current_epsilon = None
        if args.enable_dp:
            epsilon = privacy_engine.get_epsilon(delta=args.delta)
            current_epsilon = epsilon
            results['epsilon'].append(epsilon)
            logger.info(f"Current ε: {epsilon:.2f}")
        
        # Patience must monitor the same target metric used for model selection
        # in the full-mammogram utility experiments.
        if args.early_stopping_metric == 'auprc':
            if val_auprc > best_early_stopping_auprc:
                best_early_stopping_auprc = val_auprc
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Save best model based on selection criterion
        if args.select_by_auprc:
            improved = val_auprc > best_val_auprc or (val_auprc == best_val_auprc and val_loss < best_selection_loss)
        else:
            improved = val_acc > best_val_acc

        epsilon_eligible = True
        if args.enable_dp and args.dp_require_epsilon_near_target_for_selection:
            epsilon_eligible = is_epsilon_within_target(
                target_epsilon=args.epsilon,
                actual_epsilon=current_epsilon,
                lower_fraction=args.dp_selection_min_epsilon_fraction,
                upper_fraction=args.dp_selection_max_epsilon_fraction,
            )
            if not epsilon_eligible:
                logger.info(
                    "Skipping best-checkpoint update because actual epsilon "
                    f"{current_epsilon:.4f} is outside "
                    f"[{args.epsilon * args.dp_selection_min_epsilon_fraction:.4f}, "
                    f"{args.epsilon * args.dp_selection_max_epsilon_fraction:.4f}]"
                )
        
        if improved and epsilon_eligible:
            best_val_acc = val_acc
            best_val_auprc = val_auprc
            best_selection_loss = val_loss
            checkpoint_path = Path(args.output_dir) / 'best_model.pth'
            model_to_save = unwrap_model_for_state_dict(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_auprc': val_auprc,
                'epsilon': current_epsilon,
                'target_epsilon': args.epsilon if args.enable_dp else None,
                'delta': args.delta if args.enable_dp else None,
                'dp_accountant': args.dp_accountant if args.enable_dp else None,
                'noise_multiplier': getattr(optimizer, 'noise_multiplier', None) if args.enable_dp else None,
                'max_grad_norm': args.max_grad_norm if args.enable_dp else None,
                'validation_positive_class': args.validation_positive_class,
                'validation_positive_class_index': pos_idx,
                'finetune_head_strategy': args.finetune_head_strategy,
                'head_type': args.head_type,
                'image_size': args.image_size,
                'resize_mode': args.resize_mode,
                'normalization': '0.5' if args.normalize_05 else 'imagenet',
                'backbone_learning_rate': backbone_lr,
                'head_learning_rate': head_lr,
                'early_stopping_metric': args.early_stopping_metric,
                'model_type': args.model,
                'class_to_idx': training_class_to_idx,
                'train_class_counts': [int(value) for value in class_counts.tolist()],
                'train_samples': len(train_dataset),
                'validation_samples': len(val_dataset),
                'use_weighted_sampler': args.use_weighted_sampler,
                'dropout': args.dropout,
            }, checkpoint_path)
            saved_best_checkpoint = True
            criterion_name = "AUPRC" if args.select_by_auprc else "accuracy"
            logger.info(f"Best model saved (by {criterion_name}): {checkpoint_path}")
        
        # Early stopping check
        if not args.no_early_stopping and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}", flush=True)
            print(
                f"No improvement in validation {args.early_stopping_metric} "
                f"for {args.early_stopping_patience} epochs",
                flush=True,
            )
            logger.info(f"Early stopping at epoch {epoch+1}")
            results['early_stopped'] = True
            results['stopped_at_epoch'] = epoch + 1
            break
    
    if not saved_best_checkpoint:
        checkpoint_path = Path(args.output_dir) / 'best_model.pth'
        model_to_save = unwrap_model_for_state_dict(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_auprc': val_auprc,
            'epsilon': current_epsilon,
            'target_epsilon': args.epsilon if args.enable_dp else None,
            'delta': args.delta if args.enable_dp else None,
            'dp_accountant': args.dp_accountant if args.enable_dp else None,
            'noise_multiplier': getattr(optimizer, 'noise_multiplier', None) if args.enable_dp else None,
            'max_grad_norm': args.max_grad_norm if args.enable_dp else None,
            'validation_positive_class': args.validation_positive_class,
            'validation_positive_class_index': pos_idx,
            'finetune_head_strategy': args.finetune_head_strategy,
            'head_type': args.head_type,
            'image_size': args.image_size,
            'resize_mode': args.resize_mode,
            'normalization': '0.5' if args.normalize_05 else 'imagenet',
            'backbone_learning_rate': backbone_lr,
            'head_learning_rate': head_lr,
            'early_stopping_metric': args.early_stopping_metric,
            'model_type': args.model,
            'class_to_idx': training_class_to_idx,
            'train_class_counts': [int(value) for value in class_counts.tolist()],
            'train_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'use_weighted_sampler': args.use_weighted_sampler,
            'dropout': args.dropout,
            'fallback_final_checkpoint': True,
        }, checkpoint_path)
        logger.warning(f"No eligible best checkpoint was saved; saved final epoch checkpoint to {checkpoint_path}")

    # Save final results
    results_path = Path(args.output_dir) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best validation AUPRC: {best_val_auprc:.4f}")
    logger.info(f"Results saved to: {results_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
