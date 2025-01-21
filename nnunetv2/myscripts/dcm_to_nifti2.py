import os
import numpy as np
import pydicom
import nibabel as nib

def dcm_to_nii(input_folder, output_folder):
    """
    Converts a DICOM (.dcm) file to a NIfTI (.nii.gz) file.
    Parameters:
    - input_folder (str): Path to the folder containing the input .dcm file.
    - output_folder (str): Path to the folder where the output .nii.gz file will be saved.

    Returns:
    - None: The .nii.gz file is saved in the output folder.
    """
    # Find the .dcm file in the input folder
    dcm_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dcm')]

    if not dcm_files:
        raise FileNotFoundError("No .dcm file found in the input folder.")

    # Load the first .dcm file
    dcm_path = os.path.join(input_folder, dcm_files[0])
    dicom_data = pydicom.dcmread(dcm_path)

    # Extract pixel array from DICOM file
    pixel_array = dicom_data.pixel_array

    # Create an affine matrix (identity matrix as a placeholder)
    affine = np.eye(4)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(pixel_array, affine)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the NIfTI file
    output_file = os.path.join(output_folder, os.path.splitext(dcm_files[0])[0] + '.nii.gz')
    nib.save(nifti_img, output_file)

    print(f"NIfTI file saved to: {output_file}")

# Example usage:
dicom_directory = "C:/Users/Test/Desktop/Bart/Data/dcmtonifti/09-14-2014-MEDLYMPH001-mediastinallymphnodes-78073/mediastinallymphnodes-45371"
output_directory = "C:/Users/Test/Desktop/Bart/Data/dcmtonifti/09-14-2014-MEDLYMPH001-mediastinallymphnodes-78073/mediastinallymphnodes-45371"

dcm_to_nii(dicom_directory, output_directory)