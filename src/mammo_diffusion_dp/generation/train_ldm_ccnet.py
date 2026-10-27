#!/usr/bin/env python3
"""
Train LDM using Richard's ccnet approach with pretrained SD models.
Based on Richard's ccnet codebase: https://github.com/RichardObi/ccnet

Key features:
- Uses pretrained UNet2DConditionModel from HuggingFace (SD 2.1-base)
- Uses AutoencoderKL.from_pretrained for VAE
- Fine-tunes pretrained UNet (not from scratch)
- Uses CLIPTextModel for text conditioning (adapted for class conditioning)
- DDPMScheduler
- Scale factor handling (0.1 or auto-calculated)

Usage:
    python src/generation/train_ldm_ccnet.py \
        --data_path data/cbis-ddsm/train \
        --output_dir models/ldm_ccnet_mammography \
        --source_model Manojb/stable-diffusion-2-1-base \
        --scale_factor 0.1 \
        --batch_size 256 \
        --n_epochs 25 \
        --fine_tune
"""

import os
import sys
import argparse
import csv
import logging
import math
import warnings
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image, ImageOps

# Hugging Face libraries
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import CLIPTextModel, CLIPTokenizer

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train LDM using ccnet approach (pretrained models)')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data (ImageFolder format)')
    parser.add_argument('--output_dir', type=str, default='models/ldm_ccnet_mammography',
                       help='Output directory for model checkpoints')
    
    # Model
    parser.add_argument('--source_model', type=str, default='Manojb/stable-diffusion-2-1-base',
                       help='Source model for pretrained VAE, UNet, and text encoder')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size (will be resized)')
    parser.add_argument(
        '--resize_mode',
        type=str,
        default='pad',
        choices=['pad', 'squash'],
        help='How to resize training mammograms: pad preserves aspect ratio on a black canvas; squash uses direct square resize.',
    )
    
    # Training
    parser.add_argument('--n_epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                       help='Base learning rate')
    parser.add_argument('--lr_warmup_steps', type=int, default=500,
                       help='Learning rate warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument(
        '--balanced_sampling',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Sample classes with equal expected frequency. Recommended for '
            'class-imbalanced conditional generation datasets.'
        ),
    )
    parser.add_argument(
        '--conditioning_mode',
        choices=['class', 'class_view'],
        default='class',
        help=(
            'Condition on diagnosis only, or jointly on diagnosis and the '
            'mammographic view read from data_path/manifest.csv.'
        ),
    )
    parser.add_argument(
        '--conditioning_backend',
        choices=['learned', 'text'],
        default='learned',
        help=(
            'Use the historical learned condition token or frozen CLIP prompt '
            'embeddings from the source Stable Diffusion model.'
        ),
    )
    parser.add_argument(
        '--condition_dropout_prob',
        type=float,
        default=0.0,
        help=(
            'Probability of replacing a condition with a learned null token. '
            'Use a value such as 0.1 to enable classifier-free guidance at generation time.'
        ),
    )
    
    # Diffusion
    parser.add_argument('--num_train_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta_schedule', type=str, default='scaled_linear',
                       help='Beta schedule for noise scheduler')
    
    # Fine-tuning
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune pretrained UNet (default: train from scratch)')
    
    # Scaling factor
    parser.add_argument('--scale_factor', type=float, default=None,
                       help='VAE latent scaling factor. If None, auto-calculated from VAE config.')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', action='store_true',
                       help='Enable UNet gradient checkpointing for lower memory use')
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing', action='store_false',
                       help='Disable UNet gradient checkpointing')
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument('--enable_xformers', action='store_true',
                       help='Enable xFormers memory-efficient attention if installed')
    
    # Checkpointing
    parser.add_argument('--checkpoint_name', type=str, default='diffusion_best_model.pth',
                       help='Checkpoint file name')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument(
        '--warm_start_from',
        type=str,
        default=None,
        help=(
            'Initialize model weights and compatible metadata from a trusted '
            'CCNET checkpoint, while starting a fresh optimizer and epoch count.'
        ),
    )
    parser.add_argument('--eval_freq', type=int, default=10,
                       help='Evaluate every N epochs')
    parser.add_argument(
        '--save_optimizer',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Include optimizer and scheduler state in training checkpoints.',
    )
    parser.add_argument(
        '--save_final_model',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Write the duplicate model-only unet_final.pt after training.',
    )
    parser.add_argument(
        '--checkpoint_precision',
        choices=['fp32', 'fp16', 'bf16'],
        default='fp32',
        help=(
            'Floating-point precision used for model tensors written to checkpoints. '
            'fp16 substantially reduces checkpoint size and can be loaded normally for inference or warm starts.'
        ),
    )
    parser.add_argument(
        '--checkpoint_save_strategy',
        choices=['immediate', 'final'],
        default='immediate',
        help=(
            'immediate atomically writes every new best checkpoint; final keeps '
            'the best checkpoint in CPU memory and writes it once after training.'
        ),
    )
    
    # Other
    parser.add_argument('--seed', type=int, default=2,
                       help='Random seed')
    parser.add_argument('--clip_grad_norm_by', type=float, default=1.0,
                       help='Clip gradient norm')
    parser.add_argument('--snr_gamma', type=float, default=5.0,
                       help='SNR gamma for loss weighting (Richard-style ccnet uses 5.0; use 0 or negative to disable).')
    parser.add_argument(
        '--foreground_loss_weight',
        type=float,
        default=1.0,
        help='Relative diffusion-loss weight inside the breast foreground.',
    )
    parser.add_argument(
        '--background_loss_weight',
        type=float,
        default=1.0,
        help='Relative diffusion-loss weight outside the breast foreground.',
    )
    parser.add_argument(
        '--background_mask_threshold',
        type=float,
        default=6.0 / 255.0,
        help='Foreground threshold in the original [0,1] image intensity range.',
    )
    parser.add_argument(
        '--foreground_mask_dilation',
        type=int,
        default=1,
        help='Conservative foreground-mask dilation in latent pixels.',
    )
    parser.add_argument(
        '--freeze_condition_embeddings',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep learned class/view condition embeddings fixed during adaptation.',
    )
    parser.add_argument(
        '--spatial_mask_conditioning',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Condition the UNet on a binary breast silhouette as a fifth input '
            'channel. This mode must be trained from the source model rather than '
            'warm-started from a four-channel CCNET checkpoint.'
        ),
    )
    parser.add_argument(
        '--background_x0_loss_weight',
        type=float,
        default=0.0,
        help=(
            'Additional predicted-clean-latent loss outside the supplied breast '
            'mask. Recommended with --spatial_mask_conditioning.'
        ),
    )
    parser.add_argument(
        '--background_x0_max_timestep',
        type=int,
        default=500,
        help=(
            'Apply the exterior predicted-clean-latent loss only at timesteps up '
            'to this value, where x0 reconstruction is numerically stable.'
        ),
    )
    parser.add_argument(
        '--minimum_exact_black_fraction',
        type=float,
        default=0.01,
        help=(
            'Fail before training if any source image has less than this fraction '
            'of exact-zero pixels. This prevents accidentally training the clean '
            'model on gray-background images.'
        ),
    )

    return parser.parse_args()


class Stage1Wrapper(nn.Module):
    """Wrapper for VAE to handle grayscale -> RGB conversion."""
    
    def __init__(self, model: nn.Module, stack_input_to_rgb: bool = False) -> None:
        super().__init__()
        self.model = model
        self.stack_input_to_rgb = stack_input_to_rgb
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1 and self.stack_input_to_rgb:
            # Convert grayscale to RGB by stacking
            x = torch.cat((x, x, x), dim=1)
            # Encode to latent space
            posterior = self.model.encode(x).latent_dist
            z = posterior.sample()
        else:
            # Direct encoding (if already RGB)
            posterior = self.model.encode(x).latent_dist
            z = posterior.sample()
        return z


def expand_unet_for_spatial_mask(unet):
    """Add a zero-initialized silhouette channel to a pretrained SD UNet."""
    old_conv = unet.conv_in
    if old_conv.in_channels == 5:
        return unet
    if old_conv.in_channels != 4:
        raise ValueError(
            'Spatial mask conditioning expects a four-channel source UNet, '
            f'got {old_conv.in_channels} channels'
        )

    new_conv = nn.Conv2d(
        5,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4].copy_(old_conv.weight)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    unet.register_to_config(in_channels=5)
    return unet


class ClassConditionedUNet(nn.Module):
    """UNet wrapper supporting learned tokens or fixed CLIP prompt embeddings."""

    def __init__(
        self,
        unet,
        num_classes=2,
        condition_embeddings=None,
        spatial_mask_conditioning=False,
    ):
        super().__init__()
        self.unet = unet
        self.num_classes = num_classes
        self.spatial_mask_conditioning = spatial_mask_conditioning

        if condition_embeddings is None:
            cross_attention_dim = getattr(unet.config, 'cross_attention_dim', 768)
            self.class_embedding = nn.Embedding(num_classes, cross_attention_dim)
            self.register_buffer('condition_embeddings', None, persistent=True)
        else:
            if condition_embeddings.ndim != 3:
                raise ValueError(
                    'condition_embeddings must have shape '
                    '[num_conditions, sequence_length, embedding_dim]'
                )
            if condition_embeddings.shape[0] != num_classes:
                raise ValueError(
                    'condition_embeddings first dimension must equal num_classes'
                )
            expected_dim = getattr(unet.config, 'cross_attention_dim', None)
            if expected_dim is not None and condition_embeddings.shape[-1] != expected_dim:
                raise ValueError(
                    f'CLIP embedding dimension {condition_embeddings.shape[-1]} '
                    f'does not match UNet cross_attention_dim {expected_dim}'
                )
            self.class_embedding = None
            self.register_buffer(
                'condition_embeddings',
                condition_embeddings.detach().clone().float(),
                persistent=True,
            )

    def forward(
        self,
        sample,
        timestep,
        class_labels=None,
        encoder_hidden_states=None,
        spatial_mask=None,
    ):
        if self.spatial_mask_conditioning:
            if spatial_mask is None:
                raise ValueError('A spatial breast mask is required by this checkpoint')
            if spatial_mask.ndim != 4 or spatial_mask.shape[1] != 1:
                raise ValueError('spatial_mask must have shape [batch, 1, height, width]')
            if spatial_mask.shape[-2:] != sample.shape[-2:]:
                spatial_mask = F.interpolate(
                    spatial_mask.float(),
                    size=sample.shape[-2:],
                    mode='nearest',
                )
            spatial_mask = spatial_mask.to(device=sample.device, dtype=sample.dtype)
            sample = torch.cat((sample, spatial_mask), dim=1)
        elif spatial_mask is not None:
            raise ValueError('A spatial mask was supplied to a four-channel checkpoint')

        if class_labels is not None:
            if self.condition_embeddings is not None:
                encoder_hidden_states = self.condition_embeddings[class_labels]
            else:
                class_emb = self.class_embedding(class_labels)
                encoder_hidden_states = class_emb.unsqueeze(1)
            encoder_hidden_states = encoder_hidden_states.to(
                device=sample.device,
                dtype=sample.dtype,
            )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample


class ResizeContainPad:
    """Resize while preserving aspect ratio and padding to a square canvas."""

    def __init__(self, size, fill=0, method=Image.Resampling.BICUBIC):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.fill = fill
        self.method = method

    def __call__(self, image):
        contained = ImageOps.contain(image, self.size, method=self.method)
        canvas = Image.new(image.mode, self.size, color=self.fill)
        left = (self.size[0] - contained.width) // 2
        top = (self.size[1] - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas


VIEW_TO_INDEX = {'CC': 0, 'MLO': 1}


def prompts_for_conditions(condition_names):
    prompts = []
    for name in condition_names:
        is_malignant = name.startswith('is_benign_false')
        diagnosis = 'malignant' if is_malignant else 'benign'
        if name.endswith('_cc'):
            view_text = 'craniocaudal (CC) view'
        elif name.endswith('_mlo'):
            view_text = 'mediolateral oblique (MLO) view'
        else:
            view_text = 'standard view'
        prompts.append(
            'a complete full-field grayscale mammogram, '
            f'{view_text}, {diagnosis} breast finding, '
            'entire breast visible, anatomically correct medical image'
        )
    return prompts


def encode_condition_prompts(source_model, prompts, include_null_condition):
    tokenizer = CLIPTokenizer.from_pretrained(source_model, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(
        source_model,
        subfolder='text_encoder',
        torch_dtype=torch.float32,
    ).eval()
    text_encoder.requires_grad_(False)

    encoded_prompts = list(prompts)
    if include_null_condition:
        encoded_prompts.append('')
    tokenized = tokenizer(
        encoded_prompts,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
    )
    with torch.inference_mode():
        embeddings = text_encoder(tokenized.input_ids)[0].cpu().float()
    del text_encoder
    return embeddings


class ManifestConditionedImageFolder(Dataset):
    """ImageFolder dataset whose target is the joint diagnosis/view condition."""

    def __init__(self, data_path, transform):
        self.base = datasets.ImageFolder(data_path, transform=transform)
        manifest_path = Path(data_path) / 'manifest.csv'
        if not manifest_path.exists():
            raise FileNotFoundError(
                f'class_view conditioning requires a manifest: {manifest_path}'
            )

        view_by_image = {}
        with manifest_path.open('r', newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                class_name = row.get('class_name', '').strip()
                output_value = (
                    row.get('output_image', '').strip()
                    or row.get('output_path', '').strip()
                )
                view = row.get('view', '').strip().upper()
                if class_name and output_value and view in VIEW_TO_INDEX:
                    key = (class_name, Path(output_value).name)
                    if key in view_by_image and view_by_image[key] != view:
                        raise RuntimeError(
                            f'Conflicting views for {key} in {manifest_path}'
                        )
                    view_by_image[key] = view

        self.class_targets = list(self.base.targets)
        self.condition_targets = []
        missing = []
        for image_path, class_index in self.base.samples:
            class_name = self.base.classes[class_index]
            key = (class_name, Path(image_path).name)
            view = view_by_image.get(key)
            if view is None:
                missing.append(str(image_path))
                continue
            self.condition_targets.append(
                class_index * len(VIEW_TO_INDEX) + VIEW_TO_INDEX[view]
            )

        if missing:
            preview = '\n'.join(f'  - {path}' for path in missing[:10])
            raise RuntimeError(
                f'{len(missing)} images have no CC/MLO view in {manifest_path}. '
                f'First missing entries:\n{preview}'
            )

        self.condition_names = [
            f'{class_name}_{view.lower()}'
            for class_name in self.base.classes
            for view in VIEW_TO_INDEX
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, _ = self.base[index]
        return image, self.condition_targets[index]


def get_dataloader(
    data_path,
    batch_size,
    num_workers,
    image_size=256,
    resize_mode='pad',
    balanced_sampling=False,
    seed=42,
    conditioning_mode='class',
):
    """Create dataloader for mammography images."""
    resize_transform = (
        ResizeContainPad(image_size, fill=0)
        if resize_mode == 'pad'
        else transforms.Resize((image_size, image_size))
    )

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        resize_transform,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    if conditioning_mode == 'class_view':
        dataset = ManifestConditionedImageFolder(data_path, transform=transform)
        class_targets = torch.as_tensor(dataset.class_targets, dtype=torch.long)
        targets = torch.as_tensor(dataset.condition_targets, dtype=torch.long)
        class_names = dataset.base.classes
        condition_names = dataset.condition_names
    else:
        dataset = datasets.ImageFolder(data_path, transform=transform)
        class_targets = torch.as_tensor(dataset.targets, dtype=torch.long)
        targets = class_targets
        class_names = dataset.classes
        condition_names = list(dataset.classes)

    class_counts = torch.bincount(class_targets, minlength=len(class_names))
    if torch.any(class_counts == 0):
        missing = [
            class_names[index]
            for index, count in enumerate(class_counts.tolist())
            if count == 0
        ]
        raise RuntimeError(f'Cannot train with empty classes: {missing}')

    condition_counts = torch.bincount(targets, minlength=len(condition_names))
    if torch.any(condition_counts == 0):
        missing = [
            condition_names[index]
            for index, count in enumerate(condition_counts.tolist())
            if count == 0
        ]
        raise RuntimeError(f'Cannot train with empty conditions: {missing}')

    sampler = None
    if balanced_sampling:
        condition_weights = condition_counts.sum().double() / (
            len(condition_names) * condition_counts.double()
        )
        sample_weights = condition_weights[targets]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    source_class_counts = {
        class_name: int(class_counts[index])
        for index, class_name in enumerate(class_names)
    }
    source_condition_counts = {
        condition_name: int(condition_counts[index])
        for index, condition_name in enumerate(condition_names)
    }
    return dataloader, source_class_counts, source_condition_counts, condition_names


def audit_exact_black_training_data(dataset, minimum_exact_black_fraction):
    """Verify that every source image actually contains a clean black exterior."""
    base = dataset.base if isinstance(dataset, ManifestConditionedImageFolder) else dataset
    failures = []
    fractions = []
    for image_path, _ in base.samples:
        with Image.open(image_path) as image:
            grayscale = image.convert('L')
            histogram = grayscale.histogram()
            pixel_count = grayscale.width * grayscale.height
        exact_black_fraction = histogram[0] / pixel_count
        fractions.append(exact_black_fraction)
        if exact_black_fraction < minimum_exact_black_fraction:
            failures.append((image_path, exact_black_fraction))

    if failures:
        preview = '\n'.join(
            f'  - {path}: exact_black_fraction={fraction:.6f}'
            for path, fraction in failures[:10]
        )
        raise RuntimeError(
            f'{len(failures)} training images failed the exact-black input audit. '
            'Do not train the clean-background generator on this dataset. '
            f'First failures:\n{preview}'
        )
    return {
        'n_images': len(fractions),
        'minimum_exact_black_fraction_required': minimum_exact_black_fraction,
        'exact_black_fraction_min': min(fractions),
        'exact_black_fraction_mean': sum(fractions) / len(fractions),
        'exact_black_fraction_max': max(fractions),
    }


def latent_foreground_mask(
    images,
    latent_spatial_shape,
    background_threshold,
    foreground_dilation,
):
    """Extract a conservative latent breast mask from an exact-black image."""
    normalized_threshold = 2.0 * background_threshold - 1.0
    foreground = (images.float() > normalized_threshold).float()
    foreground = F.interpolate(
        foreground,
        size=latent_spatial_shape,
        mode='area',
    )
    # A low occupancy cutoff and dilation conservatively protect breast boundaries.
    foreground = (foreground > 0.05).float()
    if foreground_dilation > 0:
        kernel_size = 2 * foreground_dilation + 1
        foreground = F.max_pool2d(
            foreground,
            kernel_size=kernel_size,
            stride=1,
            padding=foreground_dilation,
        )
    return foreground


def latent_region_loss_weights(
    images,
    latent_spatial_shape,
    foreground_weight,
    background_weight,
    background_threshold,
    foreground_dilation,
):
    """Build normalized latent-space weights from an exact-black image background."""
    if foreground_weight == background_weight:
        return None

    foreground = latent_foreground_mask(
        images,
        latent_spatial_shape,
        background_threshold,
        foreground_dilation,
    )

    weights = (
        foreground * foreground_weight
        + (1.0 - foreground) * background_weight
    )
    # Preserve the overall loss scale while changing only its spatial emphasis.
    weights = weights / weights.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    return weights


def train_epoch(model, stage1, scheduler, train_loader, optimizer, lr_scheduler,
                device, scale_factor, accelerator, clip_grad_norm_by, snr_gamma,
                condition_dropout_prob=0.0, null_condition_id=None,
                foreground_loss_weight=1.0, background_loss_weight=1.0,
                background_mask_threshold=6.0 / 255.0,
                foreground_mask_dilation=1,
                spatial_mask_conditioning=False,
                background_x0_loss_weight=0.0,
                background_x0_max_timestep=500):
    """Train for one epoch."""
    model.train()
    stage1.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(progress_bar):
        with accelerator.accumulate(model):
            images = images.to(device)
            labels = labels.to(device)
            if condition_dropout_prob > 0:
                if null_condition_id is None:
                    raise RuntimeError('Condition dropout requires a null condition ID')
                drop_mask = torch.rand(labels.shape, device=device) < condition_dropout_prob
                if drop_mask.any():
                    labels = labels.clone()
                    labels[drop_mask] = null_condition_id
            
            # Encode images to latent space
            with torch.no_grad():
                vae_dtype = next(stage1.model.parameters()).dtype
                latents = stage1(images.to(dtype=vae_dtype))
                latents = latents * scale_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
            
            # Add noise to latents
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            latent_mask = None
            if spatial_mask_conditioning or background_x0_loss_weight > 0:
                latent_mask = latent_foreground_mask(
                    images,
                    latent_spatial_shape=latents.shape[-2:],
                    background_threshold=background_mask_threshold,
                    foreground_dilation=foreground_mask_dilation,
                )
            
            # Predict noise
            noise_pred = model(
                sample=noisy_latents,
                timestep=timesteps,
                class_labels=labels,
                spatial_mask=latent_mask if spatial_mask_conditioning else None,
            )

            if not torch.isfinite(noise_pred).all():
                raise FloatingPointError(
                    "Non-finite UNet output detected during ccnet training. "
                    "This usually indicates unstable latent scaling or learning dynamics."
                )

            # Keep the diffusion loss in float32, matching diffusers training practice
            # and avoiding mixed half/float gradients in backward.
            noise_pred_for_loss = noise_pred.float()
            noise_for_loss = noise.float()

            if latent_mask is not None:
                region_weights = (
                    latent_mask * foreground_loss_weight
                    + (1.0 - latent_mask) * background_loss_weight
                )
                region_weights = region_weights / region_weights.mean(
                    dim=(1, 2, 3), keepdim=True
                ).clamp_min(1e-8)
            else:
                region_weights = latent_region_loss_weights(
                    images,
                    latent_spatial_shape=latents.shape[-2:],
                    foreground_weight=foreground_loss_weight,
                    background_weight=background_loss_weight,
                    background_threshold=background_mask_threshold,
                    foreground_dilation=foreground_mask_dilation,
                )
            
            # Calculate per-sample loss so spatial and SNR weighting compose cleanly.
            loss = F.mse_loss(
                noise_pred_for_loss,
                noise_for_loss,
                reduction='none',
            )
            if region_weights is not None:
                loss = loss * region_weights
            loss = loss.mean(dim=list(range(1, len(loss.shape))))

            if snr_gamma is not None:
                # SNR-weighted loss (from diffusers training examples)
                # Compute SNR
                alphas_cumprod = scheduler.alphas_cumprod.to(device)
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])
                snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
                # Weight by SNR
                mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=0).max(dim=0)[0] / snr
                loss = loss * mse_loss_weights
            loss = loss.mean()

            if background_x0_loss_weight > 0:
                alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
                alpha_t = alphas_cumprod[timesteps].float().view(-1, 1, 1, 1)
                predicted_x0 = (
                    noisy_latents.float()
                    - torch.sqrt(1.0 - alpha_t) * noise_pred_for_loss
                ) / torch.sqrt(alpha_t).clamp_min(1e-6)
                exterior = (1.0 - latent_mask.float()).expand_as(predicted_x0)
                exterior_error = (
                    (predicted_x0 - latents.float()).square() * exterior
                ).sum(dim=(1, 2, 3)) / exterior.sum(dim=(1, 2, 3)).clamp_min(1.0)
                stable_timestep = (timesteps <= background_x0_max_timestep).float()
                if stable_timestep.sum() > 0:
                    exterior_loss = (
                        exterior_error * stable_timestep
                    ).sum() / stable_timestep.sum()
                    loss = loss + background_x0_loss_weight * exterior_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected during ccnet training: {loss.item()}. "
                    "Aborting before optimizer update."
                )
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                if clip_grad_norm_by is not None:
                    accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm_by)
            
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def checkpoint_model_state_dict(
    model: nn.Module,
    precision: str,
) -> dict[str, torch.Tensor]:
    """Create a detached CPU state dict in the requested storage precision."""
    dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }[precision]
    checkpoint_state = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        checkpoint_state[name] = tensor
    return checkpoint_state


def atomic_torch_save(payload: object, destination: Path) -> None:
    """Replace a checkpoint only after its complete temporary file is written."""
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()

    if not 0.0 <= args.condition_dropout_prob < 1.0:
        raise ValueError('--condition_dropout_prob must be in [0, 1)')
    if args.resume_from and args.warm_start_from:
        raise ValueError('--resume_from and --warm_start_from are mutually exclusive')
    if args.foreground_loss_weight <= 0 or args.background_loss_weight <= 0:
        raise ValueError('Foreground and background loss weights must be positive')
    if args.background_x0_loss_weight < 0:
        raise ValueError('--background_x0_loss_weight must be non-negative')
    if not 0.0 <= args.minimum_exact_black_fraction < 1.0:
        raise ValueError('--minimum_exact_black_fraction must be in [0, 1)')
    if not 0 <= args.background_x0_max_timestep < args.num_train_timesteps:
        raise ValueError(
            '--background_x0_max_timestep must be within the diffusion schedule'
        )
    if args.background_x0_loss_weight > 0 and not args.spatial_mask_conditioning:
        raise ValueError(
            '--background_x0_loss_weight requires --spatial_mask_conditioning'
        )
    if not 0.0 <= args.background_mask_threshold <= 1.0:
        raise ValueError('--background_mask_threshold must be in [0, 1]')
    if args.foreground_mask_dilation < 0:
        raise ValueError('--foreground_mask_dilation must be non-negative')
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"train_ccnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting ccnet-style LDM training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=output_dir,
    )
    device = accelerator.device

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load data
    logger.info("Loading data...")
    (
        train_loader,
        source_class_counts,
        source_condition_counts,
        condition_names,
    ) = get_dataloader(
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.image_size,
        args.resize_mode,
        args.balanced_sampling,
        args.seed,
        args.conditioning_mode,
    )
    logger.info(f"Source class counts: {source_class_counts}")
    logger.info(f"Source condition counts: {source_condition_counts}")
    logger.info(f"Conditioning mode: {args.conditioning_mode}")
    logger.info(f"Balanced sampling: {args.balanced_sampling}")
    exact_black_audit = None
    if args.spatial_mask_conditioning:
        logger.info('Auditing exact-black source backgrounds before model loading...')
        exact_black_audit = audit_exact_black_training_data(
            train_loader.dataset,
            args.minimum_exact_black_fraction,
        )
        logger.info('Exact-black source audit: %s', exact_black_audit)

    warm_start_checkpoint = None
    warm_start_source_epoch = None
    if args.warm_start_from:
        warm_start_path = Path(args.warm_start_from)
        if not warm_start_path.is_file():
            raise FileNotFoundError(
                f'Warm-start checkpoint does not exist: {warm_start_path}'
            )
        logger.info('Loading warm-start metadata from %s', warm_start_path)
        warm_start_checkpoint = torch.load(warm_start_path, map_location='cpu')
        if not isinstance(warm_start_checkpoint, dict) or 'model' not in warm_start_checkpoint:
            raise ValueError(
                '--warm_start_from must be a CCNET training checkpoint containing model weights'
            )
        expected_metadata = {
            'conditioning_mode': args.conditioning_mode,
            'conditioning_backend': args.conditioning_backend,
            'condition_names': condition_names,
        }
        for key, expected in expected_metadata.items():
            observed = warm_start_checkpoint.get(key)
            if observed is not None and observed != expected:
                raise ValueError(
                    f'Warm-start {key} mismatch: checkpoint={observed!r}, requested={expected!r}'
                )
        checkpoint_dropout = float(
            warm_start_checkpoint.get('condition_dropout_prob', 0.0)
        )
        checkpoint_spatial_mask = bool(
            warm_start_checkpoint.get('spatial_mask_conditioning', False)
        )
        if checkpoint_spatial_mask != args.spatial_mask_conditioning:
            raise ValueError(
                'Warm-start architecture mismatch: checkpoint '
                f'spatial_mask_conditioning={checkpoint_spatial_mask}, requested='
                f'{args.spatial_mask_conditioning}. A V9 checkpoint cannot initialize '
                'the clean-background architecture; start the first run from '
                '--source_model.'
            )
        if not math.isclose(
            checkpoint_dropout,
            args.condition_dropout_prob,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                'Warm-start condition dropout mismatch: '
                f'checkpoint={checkpoint_dropout}, requested={args.condition_dropout_prob}. '
                'Changing this value changes the condition-embedding shape.'
            )
        warm_start_source_epoch = warm_start_checkpoint.get('epoch')
        warm_start_checkpoint.pop('optimizer', None)
        warm_start_checkpoint.pop('lr_scheduler', None)
    
    # Load VAE (Stage 1)
    logger.info(f"Loading VAE from {args.source_model}...")
    if args.mixed_precision == "fp16":
        vae_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        vae_dtype = torch.bfloat16
    else:
        vae_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.source_model,
        subfolder="vae",
        torch_dtype=vae_dtype,
    )
    vae.requires_grad_(False)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    stage1 = Stage1Wrapper(model=vae, stack_input_to_rgb=True)
    stage1.eval()
    
    # Calculate scale factor
    checkpoint_scale_factor = (
        warm_start_checkpoint.get('scale_factor')
        if warm_start_checkpoint is not None
        else None
    )
    if args.scale_factor is None and checkpoint_scale_factor is not None:
        scale_factor = float(checkpoint_scale_factor)
        logger.info(
            'Using warm-start checkpoint scale_factor: %s',
            scale_factor,
        )
    elif args.scale_factor is None:
        scale_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        logger.info(
            f"Using VAE config scale_factor: {scale_factor} "
            "(for Richard-style ccnet runs, the repo commonly uses explicit --scale_factor 0.1)"
        )
    else:
        scale_factor = args.scale_factor
        if (
            checkpoint_scale_factor is not None
            and not math.isclose(
                float(checkpoint_scale_factor),
                float(scale_factor),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        ):
            raise ValueError(
                'Explicit --scale_factor conflicts with the warm-start checkpoint: '
                f'checkpoint={checkpoint_scale_factor}, requested={scale_factor}'
            )
        logger.info(f"Using provided scale_factor: {scale_factor}")
    
    # Load UNet
    if args.fine_tune:
        logger.info(f"Loading pretrained UNet from {args.source_model} for fine-tuning...")
        unet = UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet")
    else:
        logger.info("Creating UNet from config (training from scratch)...")
        # For from-scratch, we'd need a config file, but for now use pretrained
        unet = UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet")

    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        logger.info("Enabling UNet gradient checkpointing")
        unet.enable_gradient_checkpointing()

    if args.enable_xformers and hasattr(unet, "enable_xformers_memory_efficient_attention"):
        try:
            logger.info("Enabling xFormers memory-efficient attention")
            unet.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning(f"Could not enable xFormers attention: {exc}")

    if args.spatial_mask_conditioning:
        logger.info(
            'Expanding the source UNet from four to five input channels; the '
            'new breast-mask channel is initialized to exactly zero'
        )
        unet = expand_unet_for_spatial_mask(unet)

    num_conditions = len(condition_names)
    null_condition_id = (
        num_conditions if args.condition_dropout_prob > 0 else None
    )
    num_condition_embeddings = num_conditions + int(null_condition_id is not None)

    condition_prompts = None
    fixed_condition_embeddings = None
    if args.conditioning_backend == 'text':
        condition_prompts = prompts_for_conditions(condition_names)
        logger.info('Encoding CLIP condition prompts: %s', condition_prompts)
        fixed_condition_embeddings = encode_condition_prompts(
            args.source_model,
            condition_prompts,
            include_null_condition=null_condition_id is not None,
        )
        logger.info(
            'Encoded CLIP condition embedding shape: %s',
            tuple(fixed_condition_embeddings.shape),
        )
        if fixed_condition_embeddings.shape[0] != num_condition_embeddings:
            raise RuntimeError(
                'Encoded prompt count does not match condition embedding count'
            )

    # Keep the learned backend's historical class_embedding state key for
    # compatibility with existing V5/V6 checkpoints.
    model = ClassConditionedUNet(
        unet,
        num_classes=num_condition_embeddings,
        condition_embeddings=fixed_condition_embeddings,
        spatial_mask_conditioning=args.spatial_mask_conditioning,
    )
    if warm_start_checkpoint is not None:
        warm_start_state = warm_start_checkpoint.pop('model')
        model.load_state_dict(warm_start_state, strict=True)
        del warm_start_state
        warm_start_checkpoint = None
        logger.info(
            'Warm-started model weights from %s at source epoch %s; '
            'optimizer and epoch count will start fresh',
            args.warm_start_from,
            warm_start_source_epoch,
        )

    if args.freeze_condition_embeddings:
        if model.class_embedding is None:
            logger.info(
                'Condition embeddings are fixed buffers for backend %s; no parameter freeze needed',
                args.conditioning_backend,
            )
        else:
            model.class_embedding.requires_grad_(False)
            logger.info('Froze learned class/view condition embeddings')
    
    # Load scheduler
    logger.info("Loading DDPMScheduler...")
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=0.01,
    )
    
    # Setup LR scheduler
    lr_scheduler = None
    if args.lr_warmup_steps > 0:
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=len(train_loader) * args.n_epochs,
        )
    
    # Prepare with accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    stage1 = stage1.to(device=device, dtype=vae_dtype)
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if args.resume_from and Path(args.resume_from).exists():
        logger.info(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' not in checkpoint:
            raise ValueError(
                'The resume checkpoint has no optimizer state. Use it with '
                '--warm_start_from instead of --resume_from.'
            )
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Scale factor: {scale_factor}")
    logger.info(f"Fine-tuning: {args.fine_tune}")
    logger.info(f"Resize mode: {args.resize_mode}")
    logger.info(f"Balanced sampling: {args.balanced_sampling}")
    logger.info(f"Condition names: {condition_names}")
    logger.info(f"Conditioning backend: {args.conditioning_backend}")
    logger.info(f"Condition dropout probability: {args.condition_dropout_prob}")
    logger.info(f"Null condition ID: {null_condition_id}")
    effective_snr_gamma = args.snr_gamma if args.snr_gamma is not None and args.snr_gamma > 0 else None
    logger.info(f"SNR gamma: {effective_snr_gamma}")
    logger.info(
        'Region loss weights: foreground=%s, background=%s, threshold=%s, latent dilation=%s',
        args.foreground_loss_weight,
        args.background_loss_weight,
        args.background_mask_threshold,
        args.foreground_mask_dilation,
    )
    logger.info(f"Freeze condition embeddings: {args.freeze_condition_embeddings}")
    logger.info(f"Spatial mask conditioning: {args.spatial_mask_conditioning}")
    logger.info(
        'Background x0 loss: weight=%s, max_timestep=%s',
        args.background_x0_loss_weight,
        args.background_x0_max_timestep,
    )

    stopped_early = False
    stop_reason = None
    deferred_best_checkpoint = None

    for epoch in range(start_epoch, args.n_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.n_epochs}")
        
        # Train
        try:
            avg_loss = train_epoch(
                model, stage1, scheduler, train_loader, optimizer, lr_scheduler,
                device, scale_factor, accelerator, args.clip_grad_norm_by,
                effective_snr_gamma, args.condition_dropout_prob,
                null_condition_id,
                args.foreground_loss_weight,
                args.background_loss_weight,
                args.background_mask_threshold,
                args.foreground_mask_dilation,
                args.spatial_mask_conditioning,
                args.background_x0_loss_weight,
                args.background_x0_max_timestep,
            )
        except FloatingPointError as exc:
            stopped_early = True
            stop_reason = str(exc)
            logger.warning(
                "Stopping ccnet training early at epoch %s because of a non-finite batch: %s",
                epoch + 1,
                exc,
            )
            if best_loss < float("inf"):
                logger.warning(
                    "A best checkpoint already exists with loss %.4f; keeping that checkpoint and ending training cleanly.",
                    best_loss,
                )
                break
            raise
        
        logger.info(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

        if not math.isfinite(avg_loss):
            raise RuntimeError(
                f"Non-finite average loss at epoch {epoch+1}. "
                "Training aborted before checkpoint save."
            )

        unwrapped_model = accelerator.unwrap_model(model)
        current_checkpoint = {
            'model': checkpoint_model_state_dict(
                unwrapped_model,
                args.checkpoint_precision,
            ),
            'epoch': epoch + 1,
            'best_loss': best_loss,
            'scale_factor': scale_factor,
            'source_model': args.source_model,
            'fine_tune': args.fine_tune,
            'balanced_sampling': args.balanced_sampling,
            'source_class_counts': source_class_counts,
            'source_condition_counts': source_condition_counts,
            'conditioning_mode': args.conditioning_mode,
            'conditioning_backend': args.conditioning_backend,
            'condition_names': condition_names,
            'condition_prompts': condition_prompts,
            'num_conditions': num_conditions,
            'num_condition_embeddings': num_condition_embeddings,
            'condition_dropout_prob': args.condition_dropout_prob,
            'null_condition_id': null_condition_id,
            'warm_start_from': args.warm_start_from,
            'warm_start_source_epoch': warm_start_source_epoch,
            'checkpoint_precision': args.checkpoint_precision,
            'foreground_loss_weight': args.foreground_loss_weight,
            'background_loss_weight': args.background_loss_weight,
            'background_mask_threshold': args.background_mask_threshold,
            'foreground_mask_dilation': args.foreground_mask_dilation,
            'freeze_condition_embeddings': args.freeze_condition_embeddings,
            'generator_architecture': (
                'ccnet_spatial_mask_v1'
                if args.spatial_mask_conditioning
                else 'ccnet_v1'
            ),
            'spatial_mask_conditioning': args.spatial_mask_conditioning,
            'unet_in_channels': int(unwrapped_model.unet.config.in_channels),
            'background_x0_loss_weight': args.background_x0_loss_weight,
            'background_x0_max_timestep': args.background_x0_max_timestep,
            'exact_black_training_audit': exact_black_audit,
        }
        if args.save_optimizer:
            current_checkpoint['optimizer'] = optimizer.state_dict()
        if args.save_optimizer and lr_scheduler is not None:
            current_checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            current_checkpoint['best_loss'] = best_loss
            checkpoint_path = output_dir / args.checkpoint_name
            if args.checkpoint_save_strategy == 'final':
                deferred_best_checkpoint = current_checkpoint
                logger.info(
                    'Retained best epoch %s checkpoint in CPU memory for final save',
                    epoch + 1,
                )
            else:
                atomic_torch_save(current_checkpoint, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (
            args.checkpoint_save_strategy == 'immediate'
            and (epoch + 1) % args.eval_freq == 0
        ):
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            atomic_torch_save(current_checkpoint, checkpoint_path)

    if args.checkpoint_save_strategy == 'final':
        if deferred_best_checkpoint is None:
            raise RuntimeError('Training completed without a finite checkpoint to save')
        checkpoint_path = output_dir / args.checkpoint_name
        atomic_torch_save(deferred_best_checkpoint, checkpoint_path)
        logger.info(
            'Saved deferred best checkpoint from epoch %s to %s',
            deferred_best_checkpoint['epoch'],
            checkpoint_path,
        )
    
    # Save final model
    if args.save_final_model:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = output_dir / "unet_final.pt"
        torch.save(unwrapped_model.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")
    else:
        logger.info('Skipping duplicate final model save')
    
    # Save training info
    training_info = {
        'scale_factor': scale_factor,
        'source_model': args.source_model,
        'fine_tune': args.fine_tune,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'balanced_sampling': args.balanced_sampling,
        'source_class_counts': source_class_counts,
        'source_condition_counts': source_condition_counts,
        'conditioning_mode': args.conditioning_mode,
        'conditioning_backend': args.conditioning_backend,
        'condition_names': condition_names,
        'condition_prompts': condition_prompts,
        'num_conditions': num_conditions,
        'num_condition_embeddings': num_condition_embeddings,
        'condition_dropout_prob': args.condition_dropout_prob,
        'null_condition_id': null_condition_id,
        'warm_start_from': args.warm_start_from,
        'warm_start_source_epoch': warm_start_source_epoch,
        'save_optimizer': args.save_optimizer,
        'save_final_model': args.save_final_model,
        'checkpoint_precision': args.checkpoint_precision,
        'checkpoint_save_strategy': args.checkpoint_save_strategy,
        'foreground_loss_weight': args.foreground_loss_weight,
        'background_loss_weight': args.background_loss_weight,
        'background_mask_threshold': args.background_mask_threshold,
        'foreground_mask_dilation': args.foreground_mask_dilation,
        'freeze_condition_embeddings': args.freeze_condition_embeddings,
        'generator_architecture': (
            'ccnet_spatial_mask_v1'
            if args.spatial_mask_conditioning
            else 'ccnet_v1'
        ),
        'spatial_mask_conditioning': args.spatial_mask_conditioning,
        'unet_in_channels': 5 if args.spatial_mask_conditioning else 4,
        'background_x0_loss_weight': args.background_x0_loss_weight,
        'background_x0_max_timestep': args.background_x0_max_timestep,
        'exact_black_training_audit': exact_black_audit,
        'best_loss': best_loss,
        'vae_model': args.source_model,
        'training_date': datetime.now().isoformat(),
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
    }
    info_path = output_dir / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    logger.info(f"Saved training info to {info_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

