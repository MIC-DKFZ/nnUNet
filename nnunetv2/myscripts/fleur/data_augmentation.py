import os
import random
from PIL import Image

def apply_transformations(image, label):
    # Random horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    # Random vertical flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation
    rotation_angle = random.randint(-30, 30)  # Random angle between -30 and 30 degrees
    image = image.rotate(rotation_angle)
    label = label.rotate(rotation_angle)

    return image, label

def augment_images(input_dir, label_dir, output_image_dir, output_label_dir, num_augmented_images=10):
    # Create output directories if they do not exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Get the list of image paths
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith('.png')]
    label_paths = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith('.png')]

    for idx, image_path in enumerate(image_paths):
        original_image = Image.open(image_path).convert("L")  # Ensure the image is grayscale
        corresponding_label = Image.open(label_paths[idx]).convert("L")  # Load the corresponding label

        for i in range(num_augmented_images):
            # Apply the same transformations to both the image and label
            augmented_image, augmented_label = apply_transformations(original_image, corresponding_label)

            # Create the output filenames
            output_image_filename = f"image_{idx+1:03d}_{i:04d}.png"
            output_label_filename = f"label_{idx+1:03d}_{i:04d}.png"
            output_image_path = os.path.join(output_image_dir, output_image_filename)
            output_label_path = os.path.join(output_label_dir, output_label_filename)

            # Save the augmented images and labels
            augmented_image.save(output_image_path)
            augmented_label.save(output_label_path)
            print(f"Saved augmented image: {output_image_path}")
            print(f"Saved augmented label: {output_label_path}")

# Example usage
image_directory = r"C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset006\imagesTr"  # Image directory
label_directory = r"C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset006\labelsTr"  # Label directory
output_image_directory = r"C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset006\augmented_images"  # Output images directory
output_label_directory = r"C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset006\augmented_labels"  # Output labels directory

augment_images(image_directory, label_directory, output_image_directory, output_label_directory)
