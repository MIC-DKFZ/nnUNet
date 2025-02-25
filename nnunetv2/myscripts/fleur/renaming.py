import os

# Define the directory
labels_dir = r"C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset006\labelsTr"

# Get all .png files in the directory
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]

# Rename files by replacing "label" with "image"
for file_name in label_files:
    # Replace "label" with "image" in the filename
    new_name = file_name.replace("label", "image")
    
    # Get the full path for the old and new filenames
    old_file_path = os.path.join(labels_dir, file_name)
    new_file_path = os.path.join(labels_dir, new_name)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)

print("Files have been renamed successfully.")
