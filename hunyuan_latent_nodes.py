"""
HunyuanImage-3.0 ComfyUI Custom Nodes - Latent Control Nodes
Provides latent/noise control for composition, img2img, and noise shaping.

Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025-2026 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at [eric@historic.camera, eric@rollei.us] for licensing options.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT.

Note: HunyuanImage-3.0 model is subject to Tencent's Apache 2.0 license.
"""

import gc
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ComfyUI utilities (optional import — allows running outside ComfyUI for tests)
# ---------------------------------------------------------------------------
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

# ---------------------------------------------------------------------------
# Internal imports (same try/except pattern as the rest of the node set)
# ---------------------------------------------------------------------------
try:
    from .hunyuan_cache_v2 import get_cache, CachedModel
    from .hunyuan_shared import (
        get_available_hunyuan_models,
        resolve_hunyuan_model_path,
    )
    from .hunyuan_unified_v2 import (
        RESOLUTION_PRESETS, RESOLUTION_LIST, parse_resolution,
        HunyuanUnifiedV2, detect_quant_type,
    )
except ImportError:
    from hunyuan_cache_v2 import get_cache, CachedModel
    from hunyuan_shared import (
        get_available_hunyuan_models,
        resolve_hunyuan_model_path,
    )
    from hunyuan_unified_v2 import (
        RESOLUTION_PRESETS, RESOLUTION_LIST, parse_resolution,
        HunyuanUnifiedV2, detect_quant_type,
    )


# ---------------------------------------------------------------------------
# Constants — from model.config (verified against HunyuanImage-3 config.json)
# ---------------------------------------------------------------------------
# These defaults are used by HunyuanEmptyLatent which doesn't require a loaded
# model. HunyuanGenerateWithLatent validates against the actual model config.
DEFAULT_LATENT_CHANNELS = 32        # model.config.vae["latent_channels"]
DEFAULT_VAE_DOWNSAMPLE = (16, 16)   # model.config.vae_downsample_factor
VAE_SCALING_FACTOR = 0.562679178327931  # model.config.vae["scaling_factor"]


# ============================================================================
#  Node 1: HunyuanEmptyLatent
# ============================================================================

class HunyuanEmptyLatent:
    """
    Creates a random noise latent tensor at a specific resolution and seed.
    
    The output matches the exact shape the HunyuanImage-3 pipeline expects
    for its latent input: (batch, 32, H//16, W//16) in bfloat16.
    
    When connected to HunyuanGenerateWithLatent with the same seed and
    resolution, the output should be identical to the standard V2 node,
    confirming that the pipeline correctly uses the provided latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (RESOLUTION_LIST, {"default": "1024x1024 (1:1 Square)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "Eric/HunyuanImage3/Latent"

    def generate(self, resolution: str, seed: int, batch_size: int):
        image_size_str, target_h, target_w = parse_resolution(resolution)

        if target_h is None or target_w is None:
            # "auto" resolution — default to 1024x1024
            target_h, target_w = 1024, 1024
            logger.warning("HunyuanEmptyLatent: 'Auto' resolution not supported, defaulting to 1024x1024")

        latent_h = target_h // DEFAULT_VAE_DOWNSAMPLE[0]
        latent_w = target_w // DEFAULT_VAE_DOWNSAMPLE[1]
        shape = (batch_size, DEFAULT_LATENT_CHANNELS, latent_h, latent_w)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cpu")

        logger.info(
            f"HunyuanEmptyLatent: shape={shape}, seed={seed}, "
            f"pixel_size={target_h}x{target_w}"
        )

        return ({
            "latent": latent,
            "height": target_h,
            "width": target_w,
            "seed": seed,
        },)


# ============================================================================
#  Node 3: HunyuanLatentNoise
# ============================================================================

NOISE_OPERATIONS = [
    "low_pass_filter",
    "high_pass_filter",
    "blend_seeds",
    "spatial_mask_left_right",
    "spatial_mask_top_bottom",
    "amplify_center",
    "invert",
]


class HunyuanLatentNoise:
    """
    Applies noise-shaping transformations to an existing Hunyuan latent.
    Use these to influence composition without changing the prompt.
    
    Operations:
      - low_pass_filter:  Gaussian blur → stronger macro composition, less micro detail
      - high_pass_filter: Remove low frequencies → more detail variation, less composition
      - blend_seeds:      Slerp between current latent and a new seed's noise
      - spatial_mask_left_right:  Different seed noise in left vs right halves
      - spatial_mask_top_bottom:  Different seed noise in top vs bottom halves
      - amplify_center:   Boost noise magnitude in the center region
      - invert:           Negate the noise pattern (artistic effect)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("HUNYUAN_LATENT",),
                "operation": (NOISE_OPERATIONS, {"default": "low_pass_filter"}),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "Eric/HunyuanImage3/Latent"

    def apply(self, latent: dict, operation: str, strength: float, seed: int):
        tensor = latent["latent"].clone()  # Don't mutate the input
        result = getattr(self, f"_op_{operation}")(tensor, strength, seed)

        logger.info(
            f"HunyuanLatentNoise: op={operation}, strength={strength}, "
            f"shape={tuple(result.shape)}"
        )

        return ({
            "latent": result,
            "height": latent["height"],
            "width": latent["width"],
            "seed": latent.get("seed", seed),
        },)

    # ── Operation implementations ────────────────────────────────

    @staticmethod
    def _op_low_pass_filter(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Gaussian blur the noise in spatial dimensions."""
        if strength <= 0.0:
            return tensor
        # Kernel size scales with strength: 3 at low, up to 15 at high
        ksize = max(3, int(3 + strength * 12))
        if ksize % 2 == 0:
            ksize += 1  # Must be odd

        # Apply per-channel Gaussian blur via separable convolution
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Reshape for depthwise conv: (C, 1, k) and (C, 1, 1, k)
        b, c, h, w = tensor.shape
        t = tensor.float()

        # Horizontal pass
        kh = kernel_1d.view(1, 1, 1, ksize).expand(c, 1, 1, ksize)
        t = torch.nn.functional.conv2d(
            t.view(b * c, 1, h, w),
            kh.view(c, 1, 1, ksize)[:1].expand(1, 1, 1, ksize),
            padding=(0, ksize // 2),
        ).view(b, c, h, w)

        # Vertical pass
        kv = kernel_1d.view(1, 1, ksize, 1).expand(c, 1, ksize, 1)
        t = torch.nn.functional.conv2d(
            t.view(b * c, 1, h, w),
            kv.view(c, 1, ksize, 1)[:1].expand(1, 1, ksize, 1),
            padding=(ksize // 2, 0),
        ).view(b, c, h, w)

        # Blend original with blurred
        result = (1.0 - strength) * tensor.float() + strength * t
        return result.to(dtype=tensor.dtype)

    @staticmethod
    def _op_high_pass_filter(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Remove low frequencies: high_pass = original - low_pass."""
        if strength <= 0.0:
            return tensor
        low = HunyuanLatentNoise._op_low_pass_filter(tensor, 1.0, seed)
        high = tensor.float() - low.float()
        result = (1.0 - strength) * tensor.float() + strength * high
        return result.to(dtype=tensor.dtype)

    @staticmethod
    def _op_blend_seeds(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Spherical interpolation (slerp) between current latent and new noise."""
        if strength <= 0.0:
            return tensor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        other = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype, device="cpu")

        # Slerp in flattened space
        a = tensor.float().flatten()
        b = other.float().flatten()
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        dot = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            # Nearly parallel — linear interpolation
            result = (1.0 - strength) * a + strength * b
        else:
            sin_omega = torch.sin(omega)
            result = (
                torch.sin((1.0 - strength) * omega) / sin_omega * a
                + torch.sin(strength * omega) / sin_omega * b
            )
        return result.view(tensor.shape).to(dtype=tensor.dtype)

    @staticmethod
    def _op_spatial_mask_left_right(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Replace right half with noise from a different seed."""
        if strength <= 0.0:
            return tensor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        other = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype, device="cpu")

        _, _, _, w = tensor.shape
        mid = w // 2
        result = tensor.clone()
        # Blend the right half with the alternate noise
        result[:, :, :, mid:] = (1.0 - strength) * tensor[:, :, :, mid:] + strength * other[:, :, :, mid:]
        return result

    @staticmethod
    def _op_spatial_mask_top_bottom(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Replace bottom half with noise from a different seed."""
        if strength <= 0.0:
            return tensor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        other = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype, device="cpu")

        _, _, h, _ = tensor.shape
        mid = h // 2
        result = tensor.clone()
        result[:, :, mid:, :] = (1.0 - strength) * tensor[:, :, mid:, :] + strength * other[:, :, mid:, :]
        return result

    @staticmethod
    def _op_amplify_center(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Boost noise magnitude in the center region with a radial falloff."""
        if strength <= 0.0:
            return tensor
        _, _, h, w = tensor.shape
        # Create radial mask: 1.0 at center, 0.0 at edges
        yy = torch.linspace(-1, 1, h)
        xx = torch.linspace(-1, 1, w)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        dist = torch.sqrt(grid_x**2 + grid_y**2)
        # Gaussian-ish falloff: max boost at center
        mask = torch.exp(-dist**2 * 2.0)  # Tight center focus
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Amplify: multiply center region by (1 + strength * mask)
        amplification = 1.0 + strength * mask.to(dtype=tensor.dtype)
        return tensor * amplification

    @staticmethod
    def _op_invert(tensor: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        """Negate the noise pattern (blend toward negated version)."""
        if strength <= 0.0:
            return tensor
        result = (1.0 - strength) * tensor.float() + strength * (-tensor.float())
        return result.to(dtype=tensor.dtype)


# ============================================================================
#  Node 4: HunyuanGenerateWithLatent
# ============================================================================

# Image mode constants
IMAGE_MODES = ["composition", "img2img", "energy_map"]


class HunyuanGenerateWithLatent(HunyuanUnifiedV2):
    """
    All-in-one generation node with support for composition control, img2img,
    and custom latent injection.  Inherits ALL memory management, block swap,
    VAE placement, model loading, pipeline reset, and error recovery from
    HunyuanUnifiedV2.

    Usage modes (determined by which optional inputs are connected):
      1. No image, no latent  → identical to V2 Unified (pure text-to-image)
      2. Image only           → mode-dependent (see image_mode)
      3. Latent only          → custom noise / latent injection (from
                                 HunyuanEmptyLatent / HunyuanLatentNoise).
      4. Image + Latent       → mode-dependent, using the provided latent as
                                 the noise base instead of random noise.

    Image modes (when image is connected):
      - composition:  Extract only the broad spatial layout from the image
                      (heavy low-pass in latent space) and use it to modulate
                      noise amplitude.  No ghosting — only the macro
                      composition (where bright/dark/busy areas are) transfers.
                      Strength controls how much the layout influences noise.
      - img2img:      Traditional latent mix — (1−σ)·clean + σ·noise.
                      Low denoise preserves the image; high denoise adds more
                      random variation.  Can ghost at low denoise values.
      - energy_map:   Use the per-channel energy (absolute magnitude) of the
                      image latent to scale noise spatially.  More abstract
                      than composition — drives noise intensity from the
                      image's frequency content.

    Because model loading happens inside this node, the VAE is always
    available at the moment of encoding — no execution-order issues.
    """

    CATEGORY = "Eric/HunyuanImage3/Latent"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "final_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the full V2 Unified inputs
        base = HunyuanUnifiedV2.INPUT_TYPES()

        # Add our optional inputs
        if "optional" not in base:
            base["optional"] = {}
        base["optional"]["latent"] = ("HUNYUAN_LATENT",)
        base["optional"]["image"] = ("IMAGE",)
        base["optional"]["image_mode"] = (IMAGE_MODES, {
            "default": "composition",
            "tooltip": (
                "How the input image influences generation. "
                "composition: extracts spatial layout only (no ghosting) — "
                "use strength to control influence. "
                "img2img: traditional latent mix — can ghost at low denoise. "
                "energy_map: abstract energy-based noise modulation."
            ),
        })
        base["optional"]["denoise_strength"] = ("FLOAT", {
            "default": 0.60,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": (
                "For composition/energy_map: how strongly the image layout "
                "modulates noise (0.0 = no effect, 1.0 = maximum influence). "
                "For img2img: how much noise replaces the image "
                "(0.0 = exact reproduction, 1.0 = pure noise). "
                "Ignored when no image is connected."
            ),
        })

        return base

    @classmethod
    def IS_CHANGED(cls, force_reload: bool = False, seed: int = -1, **kwargs):
        return HunyuanUnifiedV2.IS_CHANGED(force_reload=force_reload, seed=seed, **kwargs)

    # ------------------------------------------------------------------
    # Image → Latent encoding  (runs AFTER model is loaded)
    # ------------------------------------------------------------------
    def _encode_image_to_latent(
        self,
        image: torch.Tensor,
        cached: "CachedModel",
        seed: int,
        denoise_strength: float,
        image_mode: str = "composition",
        noise_tensor: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Encode a ComfyUI IMAGE tensor into Hunyuan latent space using the
        model's VAE, then combine with noise according to *image_mode*.

        Args:
            image:            (B, H, W, 3) float32 [0, 1]
            cached:           CachedModel (must be loaded already)
            seed:             Random seed for noise generation
            denoise_strength: Controls influence (meaning depends on mode)
            image_mode:       'composition', 'img2img', or 'energy_map'
            noise_tensor:     Optional pre-shaped noise from HunyuanLatentNoise.
                              If None, random noise is generated from *seed*.

        Returns:
            dict  {latent, height, width, seed}
        """
        vae = cached.model.vae
        vae_config = vae.config

        # ── Move VAE to GPU FIRST so we can detect its real dtype/device ──
        if cached.vae_manager:
            cached.vae_manager.prepare_for_decode()  # moves VAE to GPU

        # Now read device/dtype AFTER the move (VAE manager may cast dtype)
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype
        logger.info(f"VAE encode: device={vae_device}, dtype={vae_dtype}")

        # ── Prepare image ─────────────────────────────────────────
        img = image.clone()
        if img.ndim == 3:
            img = img.unsqueeze(0)

        B, H, W, C = img.shape

        # Round to nearest dimensions divisible by 16 (VAE requirement)
        target_h = (H // 16) * 16
        target_w = (W // 16) * 16

        # Convert BHWC → BCHW
        img = img.permute(0, 3, 1, 2)  # (B, 3, H, W)

        # Resize if needed
        if target_h != H or target_w != W:
            img = torch.nn.functional.interpolate(
                img, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )
            logger.info(
                f"Resized image {H}x{W} → {target_h}x{target_w} "
                f"(nearest 16-divisible)"
            )

        # Normalize [0,1] → [-1,1] and cast to match VAE exactly
        img = img * 2.0 - 1.0
        img = img.to(device=vae_device, dtype=vae_dtype)

        # ── VAE encode (no autocast — run in VAE's native dtype) ──
        try:
            vae_result = vae.encode(img)
            if isinstance(vae_result, torch.Tensor):
                latent = vae_result
            else:
                latent = vae_result.latent_dist.sample()

            if hasattr(vae_config, "shift_factor") and vae_config.shift_factor:
                latent = latent - vae_config.shift_factor
            if hasattr(vae_config, "scaling_factor") and vae_config.scaling_factor:
                latent = latent * vae_config.scaling_factor

            # Squeeze temporal dim if present: (B, C, 1, H, W) → (B, C, H, W)
            if hasattr(vae, "ffactor_temporal") and latent.ndim == 5:
                latent = latent.squeeze(2)

        finally:
            if cached.vae_manager and cached.vae_placement == "managed":
                cached.vae_manager.cleanup_after_decode()

        # Move to CPU bfloat16 (matches pipeline expectation)
        clean_latent = latent.to(device="cpu", dtype=torch.bfloat16)

        # ── Obtain noise tensor ───────────────────────────────────
        if noise_tensor is not None and noise_tensor.shape == clean_latent.shape:
            noise = noise_tensor.to(dtype=torch.bfloat16, device="cpu")
            logger.info("Using provided latent as noise source")
        else:
            if noise_tensor is not None and noise_tensor.shape != clean_latent.shape:
                logger.warning(
                    f"Noise shape {tuple(noise_tensor.shape)} != latent "
                    f"shape {tuple(clean_latent.shape)} — generating random noise"
                )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                clean_latent.shape, generator=generator,
                dtype=torch.bfloat16, device="cpu",
            )

        # ── Apply image mode ──────────────────────────────────────
        if image_mode == "composition":
            result = self._mode_composition(clean_latent, noise, denoise_strength)
        elif image_mode == "energy_map":
            result = self._mode_energy_map(clean_latent, noise, denoise_strength)
        else:  # img2img
            result = self._mode_img2img(clean_latent, noise, denoise_strength)

        # Derive pixel dimensions from latent shape
        _, _, lh, lw = result.shape
        pixel_h = lh * DEFAULT_VAE_DOWNSAMPLE[0]
        pixel_w = lw * DEFAULT_VAE_DOWNSAMPLE[1]

        logger.info(
            f"Image → latent ({image_mode}): shape={tuple(result.shape)}, "
            f"pixel_size={pixel_h}x{pixel_w}, strength={denoise_strength}"
        )

        return {
            "latent": result,
            "height": pixel_h,
            "width": pixel_w,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Image mode implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _mode_img2img(
        clean: torch.Tensor, noise: torch.Tensor, strength: float,
    ) -> torch.Tensor:
        """Traditional img2img: linear mix of clean latent and noise."""
        if strength <= 0.0:
            return clean
        if strength >= 1.0:
            return noise
        return (1.0 - strength) * clean + strength * noise

    @staticmethod
    def _mode_composition(
        clean: torch.Tensor, noise: torch.Tensor, strength: float,
    ) -> torch.Tensor:
        """
        Composition mode: extract the broad spatial layout from the image
        latent and use it to modulate noise amplitude.

        Steps:
          1. Heavy Gaussian blur of the clean latent → macro structure only
             (no recognisable content survives, just spatial energy distribution).
          2. Normalise the blurred map to zero-mean, unit-variance per channel.
          3. Modulate noise: noise * (1 + strength * normalised_map)
             Areas that were bright/active in the image get louder noise;
             quiet/dark areas get softer noise.  The model interprets louder
             noise regions as "more stuff goes here".
          4. Re-normalise total noise energy so the scheduler sees the
             expected magnitude.
        """
        if strength <= 0.0:
            return noise

        b, c, h, w = clean.shape
        t = clean.float()

        # Heavy blur — kernel size proportional to spatial dims so the
        # blur wipes all recognisable structure regardless of latent size.
        ksize = max(7, min(h, w) // 2)
        if ksize % 2 == 0:
            ksize += 1  # must be odd
        sigma = ksize / 3.0

        x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable 2-D Gaussian applied per-sample-per-channel
        blurred = t.reshape(b * c, 1, h, w)
        kh = kernel_1d.view(1, 1, 1, ksize)
        kv = kernel_1d.view(1, 1, ksize, 1)
        blurred = torch.nn.functional.conv2d(blurred, kh, padding=(0, ksize // 2))
        blurred = torch.nn.functional.conv2d(blurred, kv, padding=(ksize // 2, 0))
        blurred = blurred.reshape(b, c, h, w)

        # Normalise per-channel to zero-mean, unit-variance
        for ci in range(c):
            ch = blurred[:, ci]
            mu = ch.mean()
            sd = ch.std().clamp(min=1e-6)
            blurred[:, ci] = (ch - mu) / sd

        # Modulate noise amplitude
        modulation = 1.0 + strength * blurred  # > 1 in bright areas, < 1 in dark
        modulated = noise.float() * modulation

        # Re-normalise to match original noise magnitude (keep scheduler happy)
        orig_rms = noise.float().pow(2).mean().sqrt().clamp(min=1e-8)
        new_rms = modulated.pow(2).mean().sqrt().clamp(min=1e-8)
        modulated = modulated * (orig_rms / new_rms)

        logger.info(
            f"Composition mode: blur_k={ksize}, sigma={sigma:.1f}, "
            f"modulation range=[{modulation.min():.2f}, {modulation.max():.2f}]"
        )
        return modulated.to(dtype=noise.dtype)

    @staticmethod
    def _mode_energy_map(
        clean: torch.Tensor, noise: torch.Tensor, strength: float,
    ) -> torch.Tensor:
        """
        Energy-map mode: use per-channel absolute magnitude of the image
        latent to scale noise spatially.

        More abstract than composition — transfers the frequency-domain
        energy distribution rather than spatial brightness.  High-energy
        latent regions (lots of detail/texture in the image) get louder
        noise, encouraging the model to put detail there.
        """
        if strength <= 0.0:
            return noise

        b, c, h, w = clean.shape
        energy = clean.float().abs()  # (B, C, H, W)

        # Smooth lightly to avoid pixel-level spikes
        smoothed = torch.nn.functional.avg_pool2d(
            energy, kernel_size=3, stride=1, padding=1,
        )

        # Normalise per-channel to [0, 1]
        for ci in range(c):
            ch = smoothed[:, ci]
            lo = ch.min()
            hi = ch.max().clamp(min=lo + 1e-6)
            smoothed[:, ci] = (ch - lo) / (hi - lo)

        # Modulate: range [1 - strength/2, 1 + strength/2]
        modulation = 1.0 + strength * (smoothed - 0.5)
        modulated = noise.float() * modulation

        # Re-normalise RMS
        orig_rms = noise.float().pow(2).mean().sqrt().clamp(min=1e-8)
        new_rms = modulated.pow(2).mean().sqrt().clamp(min=1e-8)
        modulated = modulated * (orig_rms / new_rms)

        logger.info(
            f"Energy-map mode: "
            f"modulation range=[{modulation.min():.2f}, {modulation.max():.2f}]"
        )
        return modulated.to(dtype=noise.dtype)

    # ------------------------------------------------------------------
    # Main generate entry point
    # ------------------------------------------------------------------
    def generate(
        self,
        model_name: str,
        prompt: str,
        resolution: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        blocks_to_swap: int = -1,
        vae_placement: str = "auto",
        post_action: str = "keep_loaded",
        enable_vae_tiling: bool = False,
        flow_shift: float = 2.8,
        reserve_vram_gb: float = 0.0,
        force_reload: bool = False,
        latent: Optional[dict] = None,
        image: Optional[torch.Tensor] = None,
        image_mode: str = "composition",
        denoise_strength: float = 0.60,
    ):
        """
        Generate images.  When *image* or *latent* is connected the node
        encodes / injects the latent after loading the model but before
        running inference, guaranteeing the VAE is available.
        """
        import time as _time

        # ── Case 1: nothing connected → full parent delegation ────
        if image is None and latent is None:
            return super().generate(
                model_name=model_name,
                prompt=prompt,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                blocks_to_swap=blocks_to_swap,
                vae_placement=vae_placement,
                post_action=post_action,
                enable_vae_tiling=enable_vae_tiling,
                flow_shift=flow_shift,
                reserve_vram_gb=reserve_vram_gb,
                force_reload=force_reload,
            )

        # ── Cases 2-4: image and/or latent connected ─────────────
        start_time = _time.time()
        device = "cuda:0"

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()

        # Force reload handling (same as parent)
        if force_reload:
            logger.info("Force reload requested - clearing cache and VRAM...")
            try:
                status = self.cache.get_status()
                if status.get("cached"):
                    cached = self.cache.get(status["model_path"], status["quant_type"])
                    if cached and cached.block_swap_manager:
                        if hasattr(cached.block_swap_manager, "cleanup"):
                            cached.block_swap_manager.cleanup()
                            cached.block_swap_manager = None
                        elif cached.block_swap_manager.hooks_installed:
                            cached.block_swap_manager.remove_hooks()
                    self.cache.full_unload()
                gc.collect()
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                try:
                    from .hunyuan_shared import force_windows_memory_release
                except ImportError:
                    from hunyuan_shared import force_windows_memory_release
                force_windows_memory_release()
            except Exception as e:
                logger.warning(f"Force reload cleanup error: {e}")

        # Auto-detect quant type
        quant_type = detect_quant_type(model_name)

        # INT8/BF16 constraints
        if quant_type == "int8":
            if blocks_to_swap != 0:
                blocks_to_swap = 0
            if post_action == "soft_unload":
                post_action = "keep_loaded"
        elif quant_type == "bf16":
            if blocks_to_swap == 0 and post_action == "soft_unload":
                post_action = "keep_loaded"

        # Auto-enable VAE tiling for NF4/BF16
        if quant_type in ("nf4", "bf16") and not enable_vae_tiling:
            enable_vae_tiling = True

        # ── Determine resolution for VRAM planning ────────────────
        # Priority: image dimensions > latent dimensions > dropdown
        if image is not None:
            B, H, W, C = image.shape
            calc_height = (H // 16) * 16
            calc_width = (W // 16) * 16
            source = "image"
        else:
            # latent is not None (guaranteed by the guard above)
            calc_height = latent["height"]
            calc_width = latent["width"]
            source = "latent"

        image_size = f"{calc_height}x{calc_width}"

        logger.info(
            f"=== HunyuanGenerateWithLatent ===\n"
            f"  Model: {model_name} ({quant_type})\n"
            f"  Resolution: {calc_height}x{calc_width} (from {source}, "
            f"dropdown ignored)\n"
            f"  Steps: {num_inference_steps}, CFG: {guidance_scale}, "
            f"Seed: {seed}"
        )
        if image is not None:
            logger.info(f"  Mode: {image_mode} (strength={denoise_strength})")
            if latent is not None:
                logger.info("  Noise source: provided latent (not random)")
        else:
            logger.info("  Mode: latent injection")

        model_path = self._get_model_path(model_name)

        try:
            # Calculate optimal config (parent method)
            final_blocks, final_vae = self._calculate_optimal_config(
                quant_type=quant_type,
                width=calc_width,
                height=calc_height,
                blocks_to_swap=blocks_to_swap,
                vae_placement=vae_placement,
                reserve_vram_gb=reserve_vram_gb,
                device=device,
            )

            # Ensure model is loaded (parent method — handles everything)
            cached = self._ensure_model_loaded(
                model_path=model_path,
                quant_type=quant_type,
                blocks_to_swap=final_blocks,
                vae_placement=final_vae,
                device=device,
                reserve_vram_gb=reserve_vram_gb,
                width=calc_width,
                height=calc_height,
            )

            # ── Prepare the final latent tensor to inject ─────────
            if image is not None:
                # Cases 2 & 4: encode image → latent, optionally mix
                # with provided latent as noise source
                noise_src = latent["latent"] if latent is not None else None
                latent_dict = self._encode_image_to_latent(
                    image=image,
                    cached=cached,
                    seed=seed,
                    denoise_strength=denoise_strength,
                    image_mode=image_mode,
                    noise_tensor=noise_src,
                )
            else:
                # Case 3: latent only (already a dict)
                latent_dict = latent

            lt = latent_dict["latent"]
            pixel_h = latent_dict["height"]
            pixel_w = latent_dict["width"]
            image_size = f"{pixel_h}x{pixel_w}"

            # Validate latent channels
            if lt.shape[1] != DEFAULT_LATENT_CHANNELS:
                raise ValueError(
                    f"Latent channel mismatch: got {lt.shape[1]}, "
                    f"expected {DEFAULT_LATENT_CHANNELS}"
                )

            # Patch resolution group so model uses exact dimensions
            self._patch_resolution_group(cached.model, pixel_h, pixel_w)

            gen_start = _time.time()
            try:
                images, final_prompt = self._run_inference_with_latent(
                    cached=cached,
                    prompt=prompt,
                    image_size=image_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    enable_vae_tiling=enable_vae_tiling,
                    flow_shift=flow_shift,
                    latent_tensor=lt,
                )
            finally:
                self._restore_resolution_group(cached.model)
            gen_time = _time.time() - gen_start

            logger.info(f"Generation completed in {gen_time:.2f}s")
            output_tensor = self._images_to_tensor(images)

            # Post action (parent method)
            self._handle_post_action(cached, post_action, model_path, quant_type)

            total_time = _time.time() - start_time
            logger.info(f"Total time: {total_time:.2f}s")
            return (output_tensor, final_prompt)

        except torch.cuda.OutOfMemoryError as oom_error:
            logger.error(f"OOM: {oom_error}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            empty = torch.zeros((1, calc_height, calc_width, 3), dtype=torch.float32)
            return (empty, prompt)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            empty = torch.zeros((1, calc_height, calc_width, 3), dtype=torch.float32)
            return (empty, prompt)

    # ------------------------------------------------------------------
    # Inference with injected latent
    # ------------------------------------------------------------------
    def _run_inference_with_latent(
        self,
        cached: "CachedModel",
        prompt: str,
        image_size: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        enable_vae_tiling: bool,
        flow_shift: float,
        latent_tensor: torch.Tensor,
    ):
        """
        Same as the parent's _run_inference, but adds ``latents=`` to the
        gen_kwargs so the pipeline uses our tensor instead of random noise.
        """
        model = cached.model
        vae_manager = cached.vae_manager
        block_swap_manager = cached.block_swap_manager

        # VAE tiling (same as parent)
        if enable_vae_tiling:
            if hasattr(model, "enable_vae_tiling"):
                model.enable_vae_tiling()
            elif hasattr(model, "vae") and hasattr(model.vae, "enable_tiling"):
                model.vae.enable_tiling(True)
            elif hasattr(model, "vae") and hasattr(model.vae, "enable_spatial_tiling"):
                model.vae.enable_spatial_tiling(True)
                if hasattr(model.vae, "enable_temporal_tiling"):
                    model.vae.enable_temporal_tiling(True)
        else:
            if hasattr(model, "disable_vae_tiling"):
                model.disable_vae_tiling()
            elif hasattr(model, "vae") and hasattr(model.vae, "disable_tiling"):
                model.vae.disable_tiling()
            elif hasattr(model, "vae") and hasattr(model.vae, "enable_spatial_tiling"):
                model.vae.enable_spatial_tiling(False)
                if hasattr(model.vae, "enable_temporal_tiling"):
                    model.vae.enable_temporal_tiling(False)

        if vae_manager:
            vae_manager.prepare_for_decode()
        if block_swap_manager:
            block_swap_manager.reset_stats()

        try:
            # Pipeline state reset (critical — same as parent)
            if hasattr(model, "_pipeline") and model._pipeline is not None:
                pipeline = model._pipeline
                scheduler = pipeline.scheduler
                if hasattr(scheduler, "_step_index"):
                    scheduler._step_index = None
                if hasattr(scheduler, "sigmas"):
                    del scheduler.sigmas
                if hasattr(scheduler, "timesteps"):
                    scheduler.timesteps = None
                if hasattr(scheduler, "_begin_index"):
                    scheduler._begin_index = None
                if hasattr(pipeline, "model_kwargs"):
                    if "past_key_values" in pipeline.model_kwargs:
                        del pipeline.model_kwargs["past_key_values"]
                    if "output_hidden_states" in pipeline.model_kwargs:
                        del pipeline.model_kwargs["output_hidden_states"]

            if hasattr(model, "_cache"):
                model._cache = None
            if hasattr(model, "past_key_values"):
                model.past_key_values = None

            # Pre-inference memory cleanup
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Set seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Set generation config
            model.generation_config.diff_infer_steps = num_inference_steps
            model.generation_config.diff_guidance_scale = guidance_scale
            model.generation_config.flow_shift = flow_shift

            # Build gen_kwargs — injects our latent
            gen_kwargs = {
                "prompt": prompt,
                "image_size": image_size,
                "seed": seed,
                "stream": False,
                "latents": latent_tensor,  # ← injected latent
            }

            logger.info(
                f"Injecting latent: shape={tuple(latent_tensor.shape)}, "
                f"dtype={latent_tensor.dtype}"
            )

            # Progress bar
            if ProgressBar is not None:
                pbar = ProgressBar(num_inference_steps)

                def progress_callback(pipe, step, timestep, callback_kwargs):
                    pbar.update(1)
                    return callback_kwargs

                gen_kwargs["callback_on_step_end"] = progress_callback

            final_prompt = prompt

            # inference_mode decision (same as parent)
            uses_device_map = (
                cached.model_path
                and (
                    not cached.is_moveable
                    or getattr(model, "hf_device_map", None) is not None
                )
            )

            if uses_device_map:
                result = model.generate_image(**gen_kwargs)
            else:
                with torch.inference_mode():
                    result = model.generate_image(**gen_kwargs)

            if block_swap_manager and block_swap_manager.config.blocks_to_swap > 0:
                logger.info(f"Block swap stats: {block_swap_manager.stats}")

            # Handle result (same as parent)
            images = []
            if isinstance(result, Image.Image):
                images = [result]
            elif isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, Image.Image):
                        images.append(item)
                if not images and result:
                    last = result[-1] if result else None
                    if isinstance(last, Image.Image):
                        images = [last]
            elif hasattr(result, "__iter__"):
                last_frame = None
                for frame in result:
                    last_frame = frame
                if last_frame is not None:
                    images = [last_frame]

            if not images:
                raise RuntimeError("Generation returned no images")

            return images, final_prompt

        finally:
            if vae_manager and cached.vae_placement == "managed":
                vae_manager.cleanup_after_decode()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
#  Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "HunyuanEmptyLatent": HunyuanEmptyLatent,
    "HunyuanLatentNoise": HunyuanLatentNoise,
    "HunyuanGenerateWithLatent": HunyuanGenerateWithLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanEmptyLatent": "Hunyuan Empty Latent",
    "HunyuanLatentNoise": "Hunyuan Latent Noise Shaping",
    "HunyuanGenerateWithLatent": "Hunyuan Generate with Latent",
}
