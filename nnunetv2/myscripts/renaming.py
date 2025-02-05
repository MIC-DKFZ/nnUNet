import os

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

# Example usage
rename_files(r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_raw\Dataset250_LymphNodes\imagesTr")