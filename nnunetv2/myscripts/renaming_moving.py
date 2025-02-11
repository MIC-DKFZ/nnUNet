import os
import shutil

def rename_and_copy_files(input_directory, output_directory):
    """
    Renames files in a directory to the format 'MEDLN_{XXX}_0000.nii.gz', 
    copies them to a specified output directory, and renames them in the output directory.

    Parameters:
        - input_directory (str): The path to the directory containing the files to rename.
        - output_directory (str): The path to the directory where renamed files will be copied.

    Returns:
        - None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Get a sorted list of all files in the input directory
        files = sorted(f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f)))

        # Iterate through the files and rename them
        for idx, file in enumerate(files):
            # Generate the new filename with the correct format
            new_name = f"MEDLNU_{idx:03d}.nii.gz"
            
            folder_path = os.path.join(input_directory, file)
            file_name = os.listdir(folder_path)
            input_path = os.path.join(folder_path, file_name[0])

            # Full paths for the input file, temporary file in output directory, and renamed file
            temp_output_path = os.path.join(output_directory, file_name[0])
            renamed_output_path = os.path.join(output_directory, new_name)

            # Copy the file to the output directory
            shutil.copy2(input_path, temp_output_path)
            print(f"Copied: {file} -> {temp_output_path}")

            # Rename the file in the output directory
            os.rename(temp_output_path, renamed_output_path)
            print(f"Renamed in output directory: {file} -> {new_name}")

        print("Copying and renaming complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    rename_and_copy_files(
    r"C:\Users\Test\Downloads\MED_ABD_LYMPH_MASKS\MED_ABD_LYMPH_MASKS",
    r"E:\Bart\nnUNet\nnUNet_raw\Dataset250_LymphNodes\labelsTr"
    )
