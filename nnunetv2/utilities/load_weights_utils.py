from einops import rearrange
import torch.nn.functional as F
import torch
from typing import Dict, List, Tuple, Optional
import re


def adapt_conv_kernel_size(
    pretrained_weight: torch.Tensor,
    target_kernel_size: List[int],
    pretrained_kernel_size: List[int],
) -> torch.Tensor:
    """
    Adapt convolution weights when kernel sizes differ between pretrained and target.

    When a dimension changes from e.g. 3 to 1, we average the weights along that dimension.
    This preserves the learned features while making the kernel compatible.

    Args:
        pretrained_weight: Pretrained conv weight tensor of shape [out_ch, in_ch, *kernel_size]
        target_kernel_size: Target kernel size as list [D, H, W] or [H, W]
        pretrained_kernel_size: Pretrained kernel size as list [D, H, W] or [H, W]

    Returns:
        Adapted weight tensor with target kernel size
    """
    assert len(target_kernel_size) == len(pretrained_kernel_size), \
        f"Kernel dimensions must match: {len(target_kernel_size)} vs {len(pretrained_kernel_size)}"

    adapted_weight = pretrained_weight.clone()

    # Process each spatial dimension (skip out_ch and in_ch dimensions)
    for dim_idx, (target_k, pt_k) in enumerate(zip(target_kernel_size, pretrained_kernel_size)):
        spatial_dim = dim_idx + 2  # +2 to skip out_ch and in_ch dimensions

        if target_k == pt_k:
            continue  # No adaptation needed for this dimension
        elif target_k < pt_k:
            # Need to reduce kernel size: average along this dimension
            # E.g., [3,3,3] -> [1,3,3] means averaging along dim 2
            if target_k == 1:
                # Average all values along this dimension to get a single value
                adapted_weight = adapted_weight.mean(dim=spatial_dim, keepdim=True)
            else:
                # For other reductions, use adaptive avg pooling approach
                # This is a more general case, but typically we go from 3->1
                # We'll use a strided mean approach
                raise NotImplementedError(
                    f"Kernel reduction from {pt_k} to {target_k} not yet supported. "
                    "Currently only reduction to kernel size 1 is implemented."
                )
        else:
            # target_k > pt_k: Need to expand kernel size
            # This case is less common - typically we'd just use the center
            # For now, we'll pad with zeros or replicate
            raise NotImplementedError(
                f"Kernel expansion from {pt_k} to {target_k} not yet supported."
            )

    return adapted_weight


def get_stage_and_block_info_from_key(key: str) -> Optional[Tuple[int, int, str]]:
    """
    Extract stage index and block index from a state dict key.

    Expected formats:
    - "0.blocks.0.conv1.conv.weight" -> (stage=0, block=0, rest="conv1.conv.weight")
    - "1.blocks.2.conv2.norm.weight" -> (stage=1, block=2, rest="conv2.norm.weight")

    Returns:
        Tuple of (stage_idx, block_idx, rest_of_key) or None if pattern doesn't match
    """
    # Pattern for encoder stages: {stage_idx}.blocks.{block_idx}.{rest}
    match = re.match(r'^(\d+)\.blocks\.(\d+)\.(.+)$', key)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def adapt_encoder_weights_for_architecture(
    pretrained_encoder_weights: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
    pretrained_kernel_sizes: List[List[int]],
    target_kernel_sizes: List[List[int]],
    pretrained_n_stages: int,
    target_n_stages: int,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Adapt pretrained encoder weights to match a target architecture with potentially
    different kernel sizes and/or number of stages.

    Handles three cases:
    1. Kernel mismatch: Adapts conv weights by averaging along dimensions that change
    2. Pretrained deeper than target: Only loads weights for stages that exist in target
    3. Target deeper than pretrained: Loads pretrained weights for available stages,
       leaves deeper stages with random initialization

    Args:
        pretrained_encoder_weights: State dict of pretrained encoder (keys already stripped)
        target_state_dict: State dict of target encoder for shape reference
        pretrained_kernel_sizes: Kernel sizes per stage in pretrained model
        target_kernel_sizes: Kernel sizes per stage in target model
        pretrained_n_stages: Number of stages in pretrained encoder
        target_n_stages: Number of stages in target encoder
        verbose: Whether to print adaptation information

    Returns:
        Tuple of (adapted_weights, adaptation_log) where adaptation_log describes changes made
    """
    adapted_weights = {}
    adaptation_log = {}

    # Determine how many stages we can transfer
    n_stages_to_transfer = min(pretrained_n_stages, target_n_stages)

    if verbose:
        if pretrained_n_stages > target_n_stages:
            print(f"[Dynamic Adaptation] Pretrained encoder has {pretrained_n_stages} stages, "
                  f"target has {target_n_stages}. Cutting away {pretrained_n_stages - target_n_stages} deeper stages.")
        elif pretrained_n_stages < target_n_stages:
            print(f"[Dynamic Adaptation] Pretrained encoder has {pretrained_n_stages} stages, "
                  f"target has {target_n_stages}. Deeper {target_n_stages - pretrained_n_stages} stages will keep random init.")

    for key, pt_weight in pretrained_encoder_weights.items():
        # Parse the key to get stage information
        parsed = get_stage_and_block_info_from_key(key)

        if parsed is None:
            # Key doesn't match expected pattern, skip or handle differently
            if key in target_state_dict:
                adapted_weights[key] = pt_weight
            continue

        stage_idx, block_idx, rest_of_key = parsed

        # Skip if this stage is beyond what we need
        if stage_idx >= n_stages_to_transfer:
            if verbose and stage_idx == n_stages_to_transfer:
                adaptation_log[f"stage_{stage_idx}"] = "skipped (pretrained too deep)"
            continue

        # Check if the key exists in target
        if key not in target_state_dict:
            # The block might not exist in target (different n_blocks_per_stage)
            # Skip this weight
            adaptation_log[key] = "skipped (not in target architecture)"
            continue

        target_weight = target_state_dict[key]

        # Check if this is a conv weight that might need kernel adaptation
        if 'conv.weight' in rest_of_key and len(pt_weight.shape) >= 4:
            pt_kernel = list(pt_weight.shape[2:])  # [D, H, W] or [H, W]
            target_kernel = list(target_weight.shape[2:])

            if pt_kernel != target_kernel:
                # Get the kernel sizes for this stage
                pt_stage_kernel = pretrained_kernel_sizes[stage_idx] if stage_idx < len(pretrained_kernel_sizes) else pt_kernel
                target_stage_kernel = target_kernel_sizes[stage_idx] if stage_idx < len(target_kernel_sizes) else target_kernel

                # Ensure they're lists
                if isinstance(pt_stage_kernel, int):
                    pt_stage_kernel = [pt_stage_kernel] * len(pt_kernel)
                if isinstance(target_stage_kernel, int):
                    target_stage_kernel = [target_stage_kernel] * len(target_kernel)

                if verbose:
                    print(f"[Dynamic Adaptation] Adapting kernel for {key}: {pt_kernel} -> {target_kernel}")

                adapted_weight = adapt_conv_kernel_size(pt_weight, target_kernel, pt_kernel)
                adapted_weights[key] = adapted_weight
                adaptation_log[key] = f"kernel adapted: {pt_kernel} -> {target_kernel}"
                continue

        # Check shape compatibility for non-kernel adaptations
        if pt_weight.shape != target_weight.shape:
            adaptation_log[key] = f"skipped (shape mismatch: {pt_weight.shape} vs {target_weight.shape})"
            continue

        # Direct copy - shapes match
        adapted_weights[key] = pt_weight

    return adapted_weights, adaptation_log


def filter_state_dict(state_dict, skip_strings):
    found_flag = False
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if any(skip in k for skip in skip_strings):
            found_flag = True
            continue
        filtered_state_dict[k] = v

    return filtered_state_dict, found_flag

def interpolate_patch_embed_1d(patch_embed, target_len, mode="linear"):
    """Resizes patch embeddings using interpolation."""
    return F.interpolate(
        patch_embed.permute(0, 2, 1),  # [B, C, Tokens]
        size=target_len,
        mode=mode,
        align_corners=False,
    ).permute(0, 2, 1)  # [B, Tokens, C]

def interpolate_patch_embed_3d(patch_embed, in_shape, out_shape):
    """Resizes patch embeddings using 3D trilinear interpolation."""
    patch_embed = patch_embed.permute(0, 2, 1)
    patch_embed = rearrange(patch_embed, "B C (x y z) -> B C x y z", **in_shape)
    patch_embed = F.interpolate(patch_embed, size=list(out_shape.values()), mode="trilinear", align_corners=False)
    patch_embed = rearrange(patch_embed, "B C x y z -> B C (x y z)", **out_shape)
    return patch_embed.permute(0, 2, 1)

def handle_pos_embed_resize(pretrained_dict, model_dict, mode, input_shape=None, pretrained_input_patch_size=None, patch_embed_size=None):
    pretrained_pos_embed = pretrained_dict["pos_embed"]
    model_pos_embed = model_dict["pos_embed"]
    model_pos_embed_shape = model_pos_embed.shape

    # for key, value in pretrained_dict.items():
    #     print(f"{key}: {value.shape}")

    has_cls_token = "cls_token" in pretrained_dict



    if has_cls_token:
        cls_pos_embed = pretrained_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    else:
        if  "cls_token" in model_dict.keys():
            cls_pos_embed = model_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed

    if mode == "interpolate":
        resized_patch_pos_embed = interpolate_patch_embed_1d(patch_pos_embed, target_len=model_pos_embed_shape[1] - int(has_cls_token))

    elif mode == "interpolate_trilinear":
        # Calculate input/output 3D shapes
        in_shape = dict(zip("xyz", [int(d / p) for d, p in zip(pretrained_input_patch_size, patch_embed_size)]))
        out_shape = dict(zip("xyz", [int(d / p) for d, p in zip(input_shape, patch_embed_size)]))
        resized_patch_pos_embed = interpolate_patch_embed_3d(patch_pos_embed, in_shape, out_shape)

    else:
        raise NotImplementedError(f"Unknown resize mode: {mode}")
    if "cls_token" in model_dict.keys():
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
    else:
        resized_pos_embed = resized_patch_pos_embed
    pretrained_dict["pos_embed"] = resized_pos_embed