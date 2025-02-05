import nibabel as nib
import numpy as np
import os

def unify_labels(input_path, output_path):
    """
    Converts all non-zero labels in a NIfTI file to a single label (1).

    Args:
        input_path (str): Path to the input NIfTI file with multiple labels.
        output_path (str): Path to save the unified NIfTI file.

    Returns:
        None
    """
    # Load the NIfTI file
    nifti_file = nib.load(input_path)
    data = nifti_file.get_fdata()

    # Convert all non-zero labels to 1
    unified_data = np.where(data > 0, 1, 0).astype(np.uint8)

    # Create a new NIfTI image with the same affine and header
    unified_nifti = nib.Nifti1Image(unified_data, affine=nifti_file.affine, header=nifti_file.header)

    # Save the new NIfTI file
    nib.save(unified_nifti, output_path)
    print(f"Unified mask saved at: {output_path}")

def unify_dataset(input_path):

    file_names = os.listdir(input_path)
    for file_name in file_names:
        if file_name.endswith(".nii.gz"):
            input_file = os.path.join(input_path, file_name)
            output_file = os.path.join(input_path, file_name.replace("MEDLN", "MEDLNU"))
            unify_labels(input_file, output_file)



if __name__ == "__main__":
    input_path = r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_raw\Dataset250_LymphNodes\labelsTr"
    unify_dataset(input_path)