#!/usr/bin/env python3
"""
Generate synthetic images using ccnet-style LDM (pretrained UNet fine-tuned).

Based on Richard's ccnet approach with SD 2.1-base pretrained models.

Usage:
    python -m mammo_diffusion_dp.generation.generate_ldm_ccnet \
        --model_path models/ldm_ccnet_mammography/diffusion_best_model.pth \
        --output_dir data/synthetic_ldm_ccnet \
        --source_model Manojb/stable-diffusion-2-1-base \
        --num_samples_per_class 1500 \
        --num_inference_steps 50
"""

import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DDIMScheduler,
)
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

try:
    from .ldm_checkpoint_metadata import resolve_scale_factor
except ImportError:
    from ldm_checkpoint_metadata import resolve_scale_factor


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic images with ccnet LDM')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated images')
    parser.add_argument('--source_model', type=str, default='Manojb/stable-diffusion-2-1-base',
                       help='Source model for VAE and UNet config')
    
    parser.add_argument('--num_samples_per_class', type=int, default=1500,
                       help='Number of samples to generate per class')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Output image size; must be divisible by the VAE downsampling factor (8)')
    parser.add_argument('--num_inference_steps', type=int, default=100,
                       help='Number of denoising steps')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for generation')
    parser.add_argument(
        '--inference_dtype',
        choices=['float32', 'float16', 'bfloat16'],
        default='float32',
        help='UNet inference dtype. float16 is fastest on most CUDA GPUs.',
    )
    parser.add_argument(
        '--enable_xformers',
        action='store_true',
        help='Enable xFormers memory-efficient attention when installed.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Keep existing images and generate only the missing count.',
    )
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='Classifier-free guidance scale (1.0 = conditional prediction only)')
    parser.add_argument(
        '--conditioning_mode',
        choices=['auto', 'class', 'class_view'],
        default='auto',
        help='Read conditioning from the checkpoint, or override it explicitly.',
    )
    parser.add_argument(
        '--views',
        nargs='+',
        choices=['CC', 'MLO'],
        default=['CC', 'MLO'],
        help='Views to generate for a class_view checkpoint.',
    )
    
    parser.add_argument('--scale_factor', type=float, default=None,
                       help='Scale factor. Defaults to checkpoint metadata.')
    parser.add_argument(
        '--allow_scale_factor_override',
        action='store_true',
        help='Allow an explicit scale factor that conflicts with checkpoint metadata.',
    )
    parser.add_argument('--scheduler', type=str, default='ddim',
                       choices=['ddpm', 'ddim'],
                       help='Scheduler type')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument(
        '--mask_root',
        type=str,
        default=None,
        help=(
            'Root containing CC/ and MLO/ binary silhouette PNGs. Required for '
            'a ccnet_spatial_mask_v1 checkpoint.'
        ),
    )
    parser.add_argument(
        '--mask_threshold',
        type=float,
        default=0.5,
        help='Threshold used to binarize supplied silhouette masks.',
    )
    parser.add_argument(
        '--save_raw_outputs',
        action='store_true',
        help=(
            'Also save decoded images before the exact-black exterior invariant. '
            'Recommended for smoke-test diagnosis only.'
        ),
    )
    parser.add_argument(
        '--allow_mask_reuse',
        action='store_true',
        help=(
            'Permit recycling silhouettes when fewer masks than requested images '
            'are available. Keep disabled for final datasets.'
        ),
    )
    parser.add_argument(
        '--background_audit_threshold',
        type=float,
        default=6.0 / 255.0,
        help='Raw decoded intensity threshold used for exterior non-black auditing.',
    )
    
    return parser.parse_args()


class Stage1Wrapper(torch.nn.Module):
    """Wrapper for VAE to handle grayscale -> RGB conversion."""
    
    def __init__(self, model, stack_input_to_rgb=False):
        super().__init__()
        self.model = model
        self.stack_input_to_rgb = stack_input_to_rgb
    
    def decode(self, z):
        """Decode latents to images."""
        # Decode from latent space
        images = self.model.decode(z).sample
        return images


def expand_unet_for_spatial_mask(unet):
    """Rebuild the source UNet with the fifth input channel used in training."""
    old_conv = unet.conv_in
    if old_conv.in_channels == 5:
        return unet
    if old_conv.in_channels != 4:
        raise ValueError(
            'Spatial mask conditioning expects a four-channel source UNet, '
            f'got {old_conv.in_channels} channels'
        )
    new_conv = torch.nn.Conv2d(
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
    unet.conv_in = new_conv
    unet.register_to_config(in_channels=5)
    return unet


class ClassConditionedUNet(torch.nn.Module):
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
            self.class_embedding = torch.nn.Embedding(num_classes, cross_attention_dim)
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
                    f'Condition embedding dimension {condition_embeddings.shape[-1]} '
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
            if spatial_mask.shape[-2:] != sample.shape[-2:]:
                spatial_mask = torch.nn.functional.interpolate(
                    spatial_mask.float(),
                    size=sample.shape[-2:],
                    mode='nearest',
                )
            sample = torch.cat(
                (sample, spatial_mask.to(device=sample.device, dtype=sample.dtype)),
                dim=1,
            )
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


def load_view_masks(mask_root, view, image_size, threshold):
    view_dir = Path(mask_root) / view.upper()
    if not view_dir.is_dir():
        raise FileNotFoundError(f'Mask directory does not exist: {view_dir}')
    paths = sorted(
        path for path in view_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    )
    if not paths:
        raise RuntimeError(f'No silhouette images found in {view_dir}')

    masks = []
    for path in paths:
        image = Image.open(path).convert('L').resize(
            (image_size, image_size),
            Image.Resampling.NEAREST,
        )
        mask = (pil_to_tensor(image).float() / 255.0 >= threshold).float()
        fraction = float(mask.mean())
        if not 0.05 <= fraction <= 0.95:
            raise RuntimeError(
                f'Unsafe silhouette fraction {fraction:.4f} in {path}; expected [0.05, 0.95]'
            )
        masks.append((path, mask))
    return masks


def generate_sample_batches(model, vae, scheduler, condition_label, num_samples,
                            batch_size, num_inference_steps, device, scale_factor,
                            model_dtype, guidance_scale=1.0,
                            null_condition_id=None, latent_size=32,
                            latent_channels=4, view_masks=None,
                            mask_sequence_offset=0,
                            background_audit_threshold=6.0 / 255.0):
    """Yield decoded CPU batches so long runs can be saved incrementally."""
    model.eval()
    vae.eval()

    num_batches = (num_samples + batch_size - 1) // batch_size
    generated_count = 0

    with torch.inference_mode():
        for i in tqdm(range(num_batches), desc=f"Generating condition {condition_label}"):
            current_batch_size = min(batch_size, num_samples - generated_count)
            mask_paths = None
            pixel_masks = None
            latent_masks = None
            if view_masks is not None:
                selected = [
                    view_masks[(mask_sequence_offset + generated_count + j) % len(view_masks)]
                    for j in range(current_batch_size)
                ]
                mask_paths = [str(item[0]) for item in selected]
                pixel_masks = torch.stack([item[1] for item in selected]).to(device)
                latent_masks = torch.nn.functional.interpolate(
                    pixel_masks,
                    size=(latent_size, latent_size),
                    mode='area',
                )
                latent_masks = (latent_masks > 0.05).to(dtype=model_dtype)
            
            # Initialize latents with noise
            latents = torch.randn(
                (current_batch_size, latent_channels, latent_size, latent_size),
                device=device,
                dtype=model_dtype,
            )
            latents = latents * scheduler.init_noise_sigma
            
            # Create class labels
            class_labels = torch.full(
                (current_batch_size,), condition_label, dtype=torch.long, device=device
            )
            use_guidance = guidance_scale != 1.0
            if use_guidance and null_condition_id is None:
                raise RuntimeError(
                    'guidance_scale != 1 requires a checkpoint trained with '
                    '--condition_dropout_prob greater than zero'
                )
            if use_guidance:
                null_labels = torch.full(
                    (current_batch_size,),
                    null_condition_id,
                    dtype=torch.long,
                    device=device,
                )
            
            # Denoising loop
            scheduler.set_timesteps(num_inference_steps)
            
            for t in scheduler.timesteps:
                # Expand latents for classifier-free guidance if needed
                latent_model_input = scheduler.scale_model_input(latents, t)
                
                # Predict noise
                noise_pred_cond = model(
                    sample=latent_model_input,
                    timestep=t,
                    class_labels=class_labels,
                    spatial_mask=latent_masks,
                )
                if use_guidance:
                    noise_pred_uncond = model(
                        sample=latent_model_input,
                        timestep=t,
                        class_labels=null_labels,
                        spatial_mask=latent_masks,
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond
                
                # Compute previous noisy sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Scale latents before decoding
            latents_scaled = latents / scale_factor
            
            # Decode to image space
            images = vae.decode(latents_scaled.float()).sample
            
            # Convert from RGB to grayscale (take first channel or average)
            if images.shape[1] == 3:
                images = images.mean(dim=1, keepdim=True)
            
            # Denormalize from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)

            raw_images = None
            exterior_means = None
            exterior_p95 = None
            exterior_nonblack = None
            if pixel_masks is not None:
                pixel_masks = pixel_masks.to(device=images.device, dtype=images.dtype)
                exterior = 1.0 - pixel_masks
                raw_images = images.clone()
                exterior_means = []
                exterior_p95 = []
                exterior_nonblack = []
                for sample_index in range(images.shape[0]):
                    outside = raw_images[sample_index][exterior[sample_index].bool()]
                    exterior_means.append(float(outside.mean()) if outside.numel() else 0.0)
                    exterior_p95.append(
                        float(torch.quantile(outside.float(), 0.95))
                        if outside.numel() else 0.0
                    )
                    exterior_nonblack.append(
                        float((outside > background_audit_threshold).float().mean())
                        if outside.numel() else 0.0
                    )
                images = images * pixel_masks
                if torch.count_nonzero(images * exterior).item() != 0:
                    raise RuntimeError('Exact-black exterior invariant failed')
            
            yield {
                'batch_start': generated_count,
                'images': images.cpu(),
                'raw_images': raw_images.cpu() if raw_images is not None else None,
                'pixel_masks': pixel_masks.cpu() if pixel_masks is not None else None,
                'mask_paths': mask_paths,
                'raw_exterior_mean': exterior_means,
                'raw_exterior_p95': exterior_p95,
                'raw_exterior_nonblack_fraction': exterior_nonblack,
            }
            generated_count += current_batch_size
            del images, latents, latents_scaled


def existing_sample_count(destination, short_name, view_slug):
    if view_slug is None:
        pattern = f'syn_ldm_{short_name}_*.png'
    else:
        pattern = f'syn_ldm_{short_name}_{view_slug}_*.png'
    return len(list(destination.glob(pattern)))


def main():
    args = parse_args()

    if args.image_size <= 0 or args.image_size % 8 != 0:
        raise ValueError('--image_size must be a positive multiple of 8')
    if args.num_samples_per_class <= 0:
        raise ValueError('--num_samples_per_class must be positive')
    if args.guidance_scale <= 0:
        raise ValueError('--guidance_scale must be positive')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive')
    if not 0.0 <= args.background_audit_threshold <= 1.0:
        raise ValueError('--background_audit_threshold must be in [0, 1]')
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype_by_name = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    model_dtype = dtype_by_name[args.inference_dtype]
    if device.type != 'cuda' and model_dtype != torch.float32:
        print('Non-CUDA device selected; falling back to float32 inference.')
        model_dtype = torch.float32
    
    print("=" * 60)
    print("ccnet LDM Synthetic Image Generation")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Samples per class: {args.num_samples_per_class}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Image size: {args.image_size}")
    print("=" * 60)

    print("\nLoading trained checkpoint metadata...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
    model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    generator_architecture = checkpoint.get('generator_architecture', 'ccnet_v1')
    spatial_mask_conditioning = bool(
        checkpoint.get('spatial_mask_conditioning', False)
    )
    if generator_architecture == 'ccnet_spatial_mask_v1':
        spatial_mask_conditioning = True
    if spatial_mask_conditioning and args.mask_root is None:
        raise ValueError(
            'This clean-background checkpoint requires --mask_root with CC/ and MLO/ masks'
        )
    if not spatial_mask_conditioning and args.mask_root is not None:
        raise ValueError(
            '--mask_root was supplied, but this is not a spatial-mask checkpoint'
        )
    conditioning_backend = checkpoint.get('conditioning_backend', 'learned')
    if conditioning_backend == 'text':
        embedding_key = 'condition_embeddings'
    elif conditioning_backend == 'learned':
        embedding_key = 'class_embedding.weight'
    else:
        raise ValueError(
            f'Unsupported checkpoint conditioning_backend={conditioning_backend}'
        )
    if embedding_key not in model_state:
        raise KeyError(f'Checkpoint does not contain {embedding_key}')

    num_condition_embeddings = int(model_state[embedding_key].shape[0])
    checkpoint_mode = checkpoint.get('conditioning_mode', 'class')
    conditioning_mode = (
        checkpoint_mode if args.conditioning_mode == 'auto' else args.conditioning_mode
    )
    if args.conditioning_mode != 'auto' and conditioning_mode != checkpoint_mode:
        raise ValueError(
            f'Checkpoint conditioning_mode={checkpoint_mode}, but '
            f'--conditioning_mode={args.conditioning_mode} was requested'
        )

    default_condition_names = (
        [
            'is_benign_false_cc',
            'is_benign_false_mlo',
            'is_benign_true_cc',
            'is_benign_true_mlo',
        ]
        if conditioning_mode == 'class_view'
        else ['is_benign_false', 'is_benign_true']
    )
    condition_names = checkpoint.get('condition_names', default_condition_names)
    num_conditions = int(checkpoint.get('num_conditions', len(condition_names)))
    null_condition_id = checkpoint.get('null_condition_id')
    if null_condition_id is not None:
        null_condition_id = int(null_condition_id)

    if num_conditions != len(condition_names):
        raise RuntimeError(
            f'Checkpoint metadata has num_conditions={num_conditions} but '
            f'{len(condition_names)} condition names'
        )
    if num_condition_embeddings < num_conditions:
        raise RuntimeError('Checkpoint has fewer embeddings than named conditions')
    if args.guidance_scale != 1.0 and null_condition_id is None:
        raise RuntimeError(
            'This checkpoint has no unconditional embedding. Generate with '
            '--guidance_scale 1.0 or train with --condition_dropout_prob 0.1.'
        )

    print(f"Conditioning mode: {conditioning_mode}")
    print(f"Conditioning backend: {conditioning_backend}")
    print(f"Conditions: {condition_names}")
    if checkpoint.get('condition_prompts'):
        print(f"Condition prompts: {checkpoint['condition_prompts']}")
    print(f"Null condition ID: {null_condition_id}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Inference dtype: {model_dtype}")
    print(f"Resume: {args.resume}")
    print(f"Generator architecture: {generator_architecture}")
    print(f"Spatial mask conditioning: {spatial_mask_conditioning}")
    
    # Load VAE
    print("\nLoading VAE...")
    vae = AutoencoderKL.from_pretrained(args.source_model, subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded (fp32 for precision consistency)")
    
    # Resolve the latent scale from the checkpoint that produced the UNet.
    # A mismatched scale changes the decoded image distribution substantially.
    checkpoint_scale = checkpoint.get('scale_factor')
    model_dir = Path(args.model_path).parent
    training_info_path = model_dir / "training_info.json"
    training_info_scale = None
    if training_info_path.exists():
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        training_info_scale = training_info.get('scale_factor')
    scale_factor, source = resolve_scale_factor(
        checkpoint_scale=checkpoint_scale,
        training_info_scale=training_info_scale,
        explicit_scale=args.scale_factor,
        vae_scale=getattr(vae.config, "scaling_factor", 0.18215),
        allow_override=args.allow_scale_factor_override,
    )
    print(f"Using scale_factor={scale_factor} from {source}")
    
    # Load UNet
    print("\nLoading pretrained UNet...")
    unet = UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet")
    if spatial_mask_conditioning:
        unet = expand_unet_for_spatial_mask(unet)
    
    fixed_condition_embeddings = (
        model_state['condition_embeddings']
        if conditioning_backend == 'text'
        else None
    )
    model = ClassConditionedUNet(
        unet,
        num_classes=num_condition_embeddings,
        condition_embeddings=fixed_condition_embeddings,
        spatial_mask_conditioning=spatial_mask_conditioning,
    )
    
    # Load trained weights
    print("Loading trained UNet weights...")
    model.load_state_dict(model_state)
    model = model.to(device=device, dtype=model_dtype)
    if args.enable_xformers:
        try:
            model.unet.enable_xformers_memory_efficient_attention()
            print('xFormers memory-efficient attention enabled')
        except Exception as exc:
            raise RuntimeError(
                '--enable_xformers was requested but could not be enabled'
            ) from exc
    model.eval()
    print("UNet loaded")
    
    # Load scheduler
    if args.scheduler == 'ddim':
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
        )
    
    # Create output directories
    output_dir = Path(args.output_dir)
    benign_dir = output_dir / "is_benign_true"
    malignant_dir = output_dir / "is_benign_false"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / '_raw_before_exterior_zeroing'
    masks_root = output_dir / '_generation_masks'
    if args.save_raw_outputs:
        raw_root.mkdir(parents=True, exist_ok=True)
    if spatial_mask_conditioning:
        masks_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    
    class_specs = [
        ('is_benign_false', 'malignant', malignant_dir),
        ('is_benign_true', 'benign', benign_dir),
    ]
    latent_size = args.image_size // 8

    for class_name, short_name, destination in class_specs:
        if conditioning_mode == 'class_view':
            selected_views = list(dict.fromkeys(view.upper() for view in args.views))
            base_count, remainder = divmod(
                args.num_samples_per_class, len(selected_views)
            )
            generation_specs = []
            for view_index, view in enumerate(selected_views):
                condition_name = f'{class_name}_{view.lower()}'
                if condition_name not in condition_names:
                    raise RuntimeError(
                        f'Checkpoint has no condition named {condition_name}'
                    )
                generation_specs.append(
                    (
                        condition_names.index(condition_name),
                        view.lower(),
                        base_count + int(view_index < remainder),
                    )
                )
        else:
            if class_name not in condition_names:
                raise RuntimeError(f'Checkpoint has no condition named {class_name}')
            generation_specs = [
                (condition_names.index(class_name), None, args.num_samples_per_class)
            ]

        class_saved = 0
        for condition_id, view_slug, sample_count in generation_specs:
            print("\n" + "=" * 60)
            label = short_name.upper()
            if view_slug is not None:
                label += f' {view_slug.upper()}'
            print(f"Generating {label} samples (condition {condition_id})...")
            print("=" * 60)

            existing_count = (
                existing_sample_count(destination, short_name, view_slug)
                if args.resume else 0
            )
            if existing_count > sample_count:
                raise RuntimeError(
                    f'Found {existing_count} existing {label.lower()} images, '
                    f'but only {sample_count} were requested.'
                )
            remaining = sample_count - existing_count
            if existing_count:
                print(f'Resuming after {existing_count} existing {label.lower()} images')

            if remaining:
                view_masks = None
                if spatial_mask_conditioning:
                    if view_slug is None:
                        raise RuntimeError(
                            'Spatial-mask generation currently requires class_view conditioning'
                        )
                    view_masks = load_view_masks(
                        args.mask_root,
                        view_slug,
                        args.image_size,
                        args.mask_threshold,
                    )
                    if len(view_masks) < sample_count and not args.allow_mask_reuse:
                        raise RuntimeError(
                            f'Need at least {sample_count} unique {view_slug.upper()} '
                            f'masks for this run, found {len(view_masks)}. Generate '
                            'more shape-model masks or explicitly pass '
                            '--allow_mask_reuse for a non-final smoke experiment.'
                        )
                    print(
                        f'Using {len(view_masks)} diagnosis-independent '
                        f'{view_slug.upper()} silhouette masks'
                    )
                for generated in generate_sample_batches(
                    model,
                    vae,
                    scheduler,
                    condition_label=condition_id,
                    num_samples=remaining,
                    batch_size=args.batch_size,
                    num_inference_steps=args.num_inference_steps,
                    device=device,
                    scale_factor=scale_factor,
                    model_dtype=model_dtype,
                    guidance_scale=args.guidance_scale,
                    null_condition_id=null_condition_id,
                    latent_size=latent_size,
                    view_masks=view_masks,
                    mask_sequence_offset=existing_count,
                    background_audit_threshold=args.background_audit_threshold,
                ):
                    batch_start = generated['batch_start']
                    images = generated['images']
                    for batch_index, img in enumerate(images):
                        local_index = existing_count + batch_start + batch_index
                        if view_slug is None:
                            filename = f'syn_ldm_{short_name}_{local_index:04d}.png'
                        else:
                            filename = (
                                f'syn_ldm_{short_name}_{view_slug}_{local_index:04d}.png'
                            )
                        output_path = destination / filename
                        save_image(img, output_path)
                        row = {
                            'output_image': str(output_path.resolve()),
                            'class_name': class_name,
                            'view': view_slug.upper() if view_slug else '',
                            'condition_id': condition_id,
                            'seed': args.seed,
                            'mask_path': '',
                            'saved_mask': '',
                            'raw_output_image': '',
                            'raw_exterior_mean': '',
                            'raw_exterior_p95': '',
                            'raw_exterior_nonblack_fraction': '',
                            'final_exterior_exact_zero': '',
                        }
                        if generated['pixel_masks'] is not None:
                            mask_path = generated['mask_paths'][batch_index]
                            saved_mask = masks_root / filename
                            save_image(generated['pixel_masks'][batch_index], saved_mask)
                            row.update({
                                'mask_path': mask_path,
                                'saved_mask': str(saved_mask.resolve()),
                                'raw_exterior_mean': generated['raw_exterior_mean'][batch_index],
                                'raw_exterior_p95': generated['raw_exterior_p95'][batch_index],
                                'raw_exterior_nonblack_fraction': generated['raw_exterior_nonblack_fraction'][batch_index],
                                'final_exterior_exact_zero': True,
                            })
                            if args.save_raw_outputs:
                                raw_class_dir = raw_root / class_name
                                raw_class_dir.mkdir(parents=True, exist_ok=True)
                                raw_output_path = raw_class_dir / filename
                                save_image(
                                    generated['raw_images'][batch_index],
                                    raw_output_path,
                                )
                                row['raw_output_image'] = str(raw_output_path.resolve())
                        manifest_rows.append(row)
                    print(
                        f'  Saved {existing_count + batch_start + len(images)}/'
                        f'{sample_count} {label.lower()} images',
                        flush=True,
                    )
            class_saved += sample_count

        if class_saved != args.num_samples_per_class:
            raise RuntimeError(
                f'Generated {class_saved} {short_name} images, expected '
                f'{args.num_samples_per_class}'
            )

    if manifest_rows:
        manifest_path = output_dir / 'manifest.csv'
        fieldnames = sorted({key for row in manifest_rows for key in row})
        with manifest_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        report = {
            'generator_architecture': generator_architecture,
            'model_path': str(Path(args.model_path).resolve()),
            'source_model': args.source_model,
            'mask_root': str(Path(args.mask_root).resolve()) if args.mask_root else None,
            'n_images_written_this_run': len(manifest_rows),
            'outside_mask_exact_black': spatial_mask_conditioning,
            'raw_exterior_mean': (
                sum(float(row['raw_exterior_mean']) for row in manifest_rows) / len(manifest_rows)
                if spatial_mask_conditioning else None
            ),
            'raw_exterior_p95_mean': (
                sum(float(row['raw_exterior_p95']) for row in manifest_rows) / len(manifest_rows)
                if spatial_mask_conditioning else None
            ),
            'raw_exterior_nonblack_fraction_mean': (
                sum(
                    float(row['raw_exterior_nonblack_fraction'])
                    for row in manifest_rows
                ) / len(manifest_rows)
                if spatial_mask_conditioning else None
            ),
            'background_audit_threshold': args.background_audit_threshold,
            'manifest': str(manifest_path.resolve()),
        }
        with (output_dir / 'generation_report.json').open('w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    
    print("\n" + "=" * 60)
    print("Generation completed!")
    print("=" * 60)
    print(f"Benign samples: {benign_dir} ({args.num_samples_per_class} images)")
    print(f"Malignant samples: {malignant_dir} ({args.num_samples_per_class} images)")
    print(f"Total: {2 * args.num_samples_per_class} synthetic images")
    if spatial_mask_conditioning:
        print('Exterior invariant: exact black outside every supplied silhouette')
        print(f"Raw exterior audit: {output_dir / 'generation_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

