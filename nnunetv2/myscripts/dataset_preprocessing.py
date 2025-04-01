import os
import numpy as np
import pydicom
import nibabel as nib
import dicom2nifti
import shutil


def dcm_to_nii_old(input_folder, output_folder, base_name):
    """
    Converts a series of DICOM (.dcm) files representing a 3D CT scan to a single NIfTI (.nii.gz) file.

    Parameters:
    - input_folder (str): Path to the folder containing the input .dcm files.
    - output_folder (str): Path to the folder where the output .nii.gz file will be saved.
    - base_name (str): Base name for the output file, e.g., "MEDLN_{XXX}_0000.nii.gz".

    Returns:
    - str: The name of the saved .nii.gz file.
    """
    # Find all .dcm files in the input folder
    dcm_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dcm')]

    if not dcm_files:
        raise FileNotFoundError("No .dcm files found in the input folder.")

    # Load all DICOM files and sort them by slice location
    dicom_slices = []
    for dcm_file in dcm_files:
        dcm_path = os.path.join(input_folder, dcm_file)
        dicom_data = pydicom.dcmread(dcm_path)
        dicom_slices.append(dicom_data)

    # Sort slices by ImagePositionPatient (or SliceLocation as fallback)
    dicom_slices.sort(key=lambda x: getattr(x, 'ImagePositionPatient', [0])[2] if hasattr(x, 'ImagePositionPatient') else getattr(x, 'SliceLocation', 0))

    # Stack pixel arrays to create a 3D volume
    pixel_arrays = [s.pixel_array for s in dicom_slices]
    volume = np.stack(pixel_arrays, axis=-1)

    # Create an affine matrix (identity matrix as a placeholder)
    affine = np.eye(4)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check existing files and determine the next index
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(base_name.split("_{XXX}_")[0]) and f.endswith(".nii.gz")]
    next_index = len(existing_files)
    formatted_index = f"{next_index:03d}"

    # Create output file name
    output_file_name = f"{base_name.replace('{XXX}', formatted_index)}"
    output_file_path = os.path.join(output_folder, output_file_name)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)

    # Save the NIfTI file
    nib.save(nifti_img, output_file_path)

    # print(f"NIfTI file saved to: {output_file_path}")

    return output_file_name

def dcm_to_nii(input_folder, output_folder):

    dicom2nifti.convert_directory(input_folder, output_folder)

def move_and_rename_masks(mask_folder, output_mapping, destination_folder):
    """
    Moves and renames mask files based on a given mapping of folder names to file names.

    Parameters:
    - mask_folder (str): Path to the folder containing the mask files.
    - output_mapping (dict): Dictionary mapping folder names to corresponding .nii.gz file names.
    - destination_folder (str): Path to the destination folder for renamed masks.

    Returns:
    - None
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    for folder_name, file_name in output_mapping.items():
        mask_path = os.path.join(mask_folder, folder_name + "_mask.nii.gz")
        if os.path.exists(mask_path):
            new_mask_path = os.path.join(destination_folder, file_name)
            os.rename(mask_path, new_mask_path)
            print(f"Moved and renamed mask: {mask_path} -> {new_mask_path}")
        else:
            print(f"Mask not found for folder: {folder_name}")

def rename_files(directory):
    """
    Renames files in a directory to the format 'MEDLN_{XXX}.nii.gz'.

    Parameters:
        - directory (str): The path to the directory containing the files to rename.

    Returns:
        - None
    """
    try:
        # Get a sorted list of all files in the directory
        files = sorted(f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))

        # Iterate through the files and rename them
        for idx, file in enumerate(files):
            # Generate the new filename with the correct format
            new_name = f"MEDLNU_{idx:03d}_0000.nii.gz"
            
            # Full path to the current and new file names
            current_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_name)

            # Rename the file
            os.rename(current_path, new_path)
            print(f"Renamed: {file} -> {new_name}")

        print("Renaming complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

def preprocessing(input_path, output_path, dataset_name, mask_folder):
    """
    Preprocesses a dataset of DICOM files, converting them to NIfTI format and storing output names.

    Parameters:
    - input_path (str): Path to the input dataset directory.
    - output_path (str): Path to the output directory.
    - dataset_name (str): Name of the dataset for organizing outputs.

    Returns:
    - dict: A dictionary mapping folder names to their respective .nii.gz file names.
    """
    # Define output path and folder name
    images_output, labels_output = output_folder(output_path, dataset_name)

    # Get all folder names in the input path
    folder_names = [name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]

    output_mapping = {}

    # Process each folder in the input path
    for folder_name in folder_names:
        folder_path1 = os.path.join(input_path, folder_name)
        folder_path2 = os.path.join(folder_path1, os.listdir(folder_path1)[0])


        folder_names3 = [name for name in os.listdir(folder_path2) if os.path.isdir(os.path.join(folder_path2, name))]
        folder_path3 = os.path.join(folder_path2, folder_names3[1])
        
        # Create NIfTI file
        # base_name = f"MEDLN_{{XXX}}_0000.nii.gz"

        dicom2nifti.convert_directory(folder_path3, images_output)
        rename_files(images_output)



        # Store the output file name in the mapping
        # output_mapping[folder_name] = file_name

    # Move and rename mask files
    # move_and_rename_masks(mask_folder, output_mapping, labels_output)

def output_folder(output_path, folder_name):
    """
    Creates a new folder in the specified output path and subfolders for images and labels.

    Parameters:
    - output_path (str): Path to the output directory.
    - folder_name (str): Name of the new folder to be created.

    Returns:
    - tuple: Paths to the newly created images and labels subfolders.
    """
    new_folder = os.path.join(output_path, folder_name)
    os.makedirs(new_folder, exist_ok=True)
    images_output = os.path.join(new_folder, "images")
    labels_output = os.path.join(new_folder, "labels")

    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    return images_output, labels_output


input_path = r"E:\Bart\nnUNet\Raw data\manifest-1680277513580\CT Lymph Nodes"
output_path = r"E:\Bart\nnUNet\Raw data"
dataset_name = "Dataset250_LymphNodes"
masks_path = r"C:\Users\Test\Desktop\Bart\Data\Dataset CT Lymph Nodes\MED_ABD_LYMPH_MASKS\MED_ABD_LYMPH_MASKS"

if __name__ == "__main__":
    preprocessing(input_path, output_path, dataset_name, masks_path)