import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os
import argparse

def visualize_nifti_segmentation(image_path, seg_path, slice_idx=None, axis=2,
                                figsize=(15, 5), save_path=None, title_prefix=""):
    """
    Visualize NIfTI image and segmentation overlay

    Parameters:
    -----------
    image_path : str
        Path to the original NIfTI image
    seg_path : str
        Path to the segmentation NIfTI file
    slice_idx : int, optional
        Slice index to visualize. If None, uses middle slice
    axis : int
        Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
    figsize : tuple
        Figure size for matplotlib
    save_path : str, optional
        Path to save the visualization
    title_prefix : str
        Prefix for the plot title
    """

    # Load NIfTI files
    img_nii = nib.load(image_path)
    seg_nii = nib.load(seg_path)

    img_data = img_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = img_data.shape[axis] // 2

    # Extract slices based on axis
    if axis == 0:  # Sagittal
        img_slice = img_data[slice_idx, :, :]
        seg_slice = seg_data[slice_idx, :, :]
        view_name = "Sagittal"
    elif axis == 1:  # Coronal
        img_slice = img_data[:, slice_idx, :]
        seg_slice = seg_data[:, slice_idx, :]
        view_name = "Coronal"
    else:  # Axial (default)
        img_slice = img_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
        view_name = "Axial"

    # Rotate for proper anatomical orientation
    img_slice = np.rot90(img_slice)
    seg_slice = np.rot90(seg_slice)

    # Create custom colormap for segmentation
    # Assuming: 0=background, 1=pancreas, 2=lesion
    colors = ['black', 'red', 'yellow']  # Background, pancreas, lesion
    cmap_seg = ListedColormap(colors[:int(seg_slice.max()) + 1])

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    im1 = axes[0].imshow(img_slice, cmap='gray', aspect='equal')
    axes[0].set_title(f'{title_prefix}Original Image\n{view_name} - Slice {slice_idx}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Segmentation
    im2 = axes[1].imshow(seg_slice, cmap=cmap_seg, aspect='equal', vmin=0, vmax=2)
    axes[1].set_title(f'{title_prefix}Segmentation\n{view_name} - Slice {slice_idx}')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_ticks([0, 1, 2])
    cbar2.set_ticklabels(['Background', 'Pancreas', 'Lesion'])

    # Overlay
    axes[2].imshow(img_slice, cmap='gray', aspect='equal')
    # Create masked segmentation for overlay
    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    im3 = axes[2].imshow(seg_masked, cmap=cmap_seg, alpha=0.5, aspect='equal', vmin=0, vmax=2)
    axes[2].set_title(f'{title_prefix}Overlay\n{view_name} - Slice {slice_idx}')
    axes[2].axis('off')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    # plt.show()

    # Print some statistics
    unique_labels = np.unique(seg_slice)
    print(f"\nSegmentation statistics for slice {slice_idx}:")
    print(f"Unique labels in slice: {unique_labels}")
    for label in unique_labels:
        count = np.sum(seg_slice == label)
        percentage = (count / seg_slice.size) * 100
        label_name = {0: 'Background', 1: 'Pancreas', 2: 'Lesion'}.get(label, f'Label_{label}')
        print(f"{label_name}: {count} pixels ({percentage:.2f}%)")

def compute_dice_coefficient(y_true, y_pred):
    """
    Compute Dice coefficient between two binary masks

    Parameters:
    -----------
    y_true : ndarray
        Ground truth binary segmentation mask
    y_pred : ndarray
        Predicted binary segmentation mask

    Returns:
    --------
    float
        Dice coefficient between 0.0 and 1.0
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    if union == 0:
        # If both masks are empty, consider it a perfect match
        return 1.0 if np.sum(y_true) == np.sum(y_pred) == 0 else 0.0

    return (2.0 * intersection) / union

def compare_segmentations(image_path, gt_seg_path, pred_seg_path, slice_idx=None, axis=2,
                         figsize=(20, 5), save_path=None):
    """
    Compare ground truth and predicted segmentations for a given image

    Parameters:
    -----------
    image_path : str
        Path to the original NIfTI image
    gt_seg_path : str
        Path to the ground truth segmentation NIfTI file
    pred_seg_path : str
        Path to the predicted segmentation NIfTI file
    slice_idx : int, optional
        Slice index to visualize. If None, uses middle slice
    axis : int
        Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
    figsize : tuple
        Figure size for matplotlib
    save_path : str, optional
        Path to save the visualization
    """
    # Load NIfTI files
    img_nii = nib.load(image_path)
    gt_nii = nib.load(gt_seg_path)
    pred_nii = nib.load(pred_seg_path)

    img_data = img_nii.get_fdata()
    gt_data = gt_nii.get_fdata()
    pred_data = pred_nii.get_fdata()

    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = img_data.shape[axis] // 2

    # Extract slices based on axis
    if axis == 0:  # Sagittal
        img_slice = img_data[slice_idx, :, :]
        gt_slice = gt_data[slice_idx, :, :]
        pred_slice = pred_data[slice_idx, :, :]
        view_name = "Sagittal"
    elif axis == 1:  # Coronal
        img_slice = img_data[:, slice_idx, :]
        gt_slice = gt_data[:, slice_idx, :]
        pred_slice = pred_data[:, slice_idx, :]
        view_name = "Coronal"
    else:  # Axial (default)
        img_slice = img_data[:, :, slice_idx]
        gt_slice = gt_data[:, :, slice_idx]
        pred_slice = pred_data[:, :, slice_idx]
        view_name = "Axial"

    # Rotate for proper anatomical orientation
    img_slice = np.rot90(img_slice)
    gt_slice = np.rot90(gt_slice)
    pred_slice = np.rot90(pred_slice)

    # Create custom colormap for segmentation
    colors = ['black', 'red', 'yellow']  # Background, pancreas, lesion
    cmap_seg = ListedColormap(colors[:max(int(gt_slice.max()), int(pred_slice.max())) + 1])

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    im1 = axes[0].imshow(img_slice, cmap='gray', aspect='equal')
    axes[0].set_title(f'Original Image\n{view_name} - Slice {slice_idx}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Ground truth overlay
    axes[1].imshow(img_slice, cmap='gray', aspect='equal')
    gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
    im2 = axes[1].imshow(gt_masked, cmap=cmap_seg, alpha=0.5, aspect='equal', vmin=0, vmax=2)
    axes[1].set_title(f'Ground Truth\n{view_name} - Slice {slice_idx}')
    axes[1].axis('off')

    # Prediction overlay
    axes[2].imshow(img_slice, cmap='gray', aspect='equal')
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    im3 = axes[2].imshow(pred_masked, cmap=cmap_seg, alpha=0.5, aspect='equal', vmin=0, vmax=2)
    axes[2].set_title(f'Prediction\n{view_name} - Slice {slice_idx}')
    axes[2].axis('off')

    # Difference map
    diff_map = np.zeros_like(gt_slice)
    diff_map[(gt_slice > 0) & (pred_slice == 0)] = 1  # False negatives
    diff_map[(gt_slice == 0) & (pred_slice > 0)] = 2  # False positives
    diff_map[(gt_slice > 0) & (pred_slice > 0) & (gt_slice != pred_slice)] = 3  # Wrong label

    diff_colors = ['black', 'red', 'blue', 'yellow']  # Background, FN, FP, Wrong label
    diff_cmap = ListedColormap(diff_colors)

    diff_masked = np.ma.masked_where(diff_map == 0, diff_map)
    im4 = axes[3].imshow(img_slice, cmap='gray', aspect='equal')
    axes[3].imshow(diff_masked, cmap=diff_cmap, alpha=0.5, aspect='equal', vmin=0, vmax=3)
    axes[3].set_title(f'Differences\n{view_name} - Slice {slice_idx}')
    axes[3].axis('off')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to: {save_path}")

    # plt.show()

    # Compute and print metrics for this slice
    print(f"\nSegmentation comparison metrics for slice {slice_idx}:")

    # Per-class Dice scores
    for label in [1, 2]:  # Pancreas and lesion
        gt_binary = (gt_slice == label).astype(np.uint8)
        pred_binary = (pred_slice == label).astype(np.uint8)
        dice = compute_dice_coefficient(gt_binary, pred_binary)
        label_name = {1: 'Pancreas', 2: 'Lesion'}[label]
        print(f"{label_name} Dice score: {dice:.4f}")

    # Error analysis
    fn_pixels = np.sum(diff_map == 1)
    fp_pixels = np.sum(diff_map == 2)
    wrong_label_pixels = np.sum(diff_map == 3)
    total_pixels = gt_slice.size

    print(f"\nError Analysis:")
    print(f"False Negatives: {fn_pixels} pixels ({fn_pixels/total_pixels*100:.2f}%)")
    print(f"False Positives: {fp_pixels} pixels ({fp_pixels/total_pixels*100:.2f}%)")
    print(f"Wrong Labels: {wrong_label_pixels} pixels ({wrong_label_pixels/total_pixels*100:.2f}%)")

def explore_nifti_volume(image_path, seg_path, num_slices=5, axis=2, save_dir=None):
    """
    Explore multiple slices of a NIfTI volume to find interesting slices

    Parameters:
    -----------
    image_path : str
        Path to the original NIfTI image
    seg_path : str
        Path to the segmentation NIfTI file
    num_slices : int
        Number of slices to visualize
    axis : int
        Axis along which to slice
    save_dir : str, optional
        Directory to save visualizations
    """

    # Load segmentation to find slices with content
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata()

    # Find slices with segmentation content
    if axis == 0:
        slice_sums = np.sum(seg_data, axis=(1, 2))
    elif axis == 1:
        slice_sums = np.sum(seg_data, axis=(0, 2))
    else:
        slice_sums = np.sum(seg_data, axis=(0, 1))

    # Get indices of slices with content
    content_slices = np.where(slice_sums > 0)[0]

    if len(content_slices) == 0:
        print("No segmentation content found!")
        return

    # Select evenly spaced slices with content
    if len(content_slices) <= num_slices:
        selected_slices = content_slices
    else:
        step = len(content_slices) // num_slices
        selected_slices = content_slices[::step][:num_slices]

    print(f"Visualizing slices: {selected_slices}")
    print(f"Total slices with content: {len(content_slices)}")

    # Visualize selected slices
    for i, slice_idx in enumerate(selected_slices):
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"slice_{slice_idx:03d}.png")

        title_prefix = f"Slice {i+1}/{len(selected_slices)}: "
        visualize_nifti_segmentation(
            image_path, seg_path,
            slice_idx=slice_idx,
            axis=axis,
            save_path=save_path,
            title_prefix=title_prefix
        )


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NIfTI image and segmentation overlays.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the original NIfTI image')
    parser.add_argument('--gt_seg_path', type=str, required=True, help='Path to the segmentation NIfTI file')
    parser.add_argument('--pred_seg_path', type=str, default=None, help='Path to the predicted segmentation NIfTI file')
    parser.add_argument('--slice_idx', type=int, default=20, help='Slice index to visualize (default: 50)')
    parser.add_argument('--num_slices', type=int, default=5, help='Number of slices to explore (default: 5)')
    parser.add_argument('--axis', type=int, default=2, help='Axis along which to slice (0=sagittal, 1=coronal, 2=axial)')
    parser.add_argument('--save_dir', type=str, default="visualizations", help='Directory to save visualizations')

    args = parser.parse_args()

    image_path = args.image_path
    gt_seg_path = args.gt_seg_path

    # Visualize a single slice
    visualize_nifti_segmentation(image_path, gt_seg_path, slice_idx=args.slice_idx, axis=args.axis)

    # Explore multiple slices
    explore_nifti_volume(image_path, gt_seg_path, num_slices=args.num_slices, axis=args.axis, save_dir=args.save_dir)

    if args.pred_seg_path:
        # Compare ground truth and predicted segmentations
        compare_segmentations(image_path, gt_seg_path, args.pred_seg_path, slice_idx=args.slice_idx, axis=args.axis, save_path=os.path.join(args.save_dir, f"comparison_slice={args.slice_idx}.png"))
