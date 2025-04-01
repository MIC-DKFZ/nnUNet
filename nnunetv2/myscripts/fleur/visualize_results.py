import os
import matplotlib.pyplot as plt
from PIL import Image

# Define the folders containing the output predictions, ground truth labels, post-processed images, and output labels
first_result = r'C:\Users\Test\Desktop\Fleur\Run2\Validatie_set'
results_run2 = r'C:\Users\Test\Desktop\Fleur\Run2\Validatie_set\output_labels'
original_image = r'C:\Users\Test\Desktop\Fleur\Run2\results'
label_directory = r'C:\Users\Test\Desktop\Fleur\extract_boundaries\results_boundaryextraction'

# Get a list of all PNG files in the output folder
output_files = [f for f in os.listdir(first_result) if f.endswith('.png')]

# Loop through the output files and display them alongside the ground truth labels, post-processed images, and the new label images
for output_file in output_files:
    # Load the predicted image
    output_path = os.path.join(first_result, output_file)
    first_run = Image.open(output_path)

    # Load the corresponding ground truth label
    label_path = os.path.join(results_run2, output_file)  # Assuming filenames match
    second_run = Image.open(label_path)

    # Load the corresponding post-processed image
    postprocessed_path = os.path.join(original_image, output_file)  # Assuming filenames match
    original = Image.open(postprocessed_path)

    # Load the corresponding label image
    label_path = os.path.join(label_directory, output_file)  # Assuming filenames match
    label_image = Image.open(label_path)

    # Display the images side by side
    plt.figure(figsize=(24, 6))

    # Predicted Image
    plt.subplot(1, 4, 1)
    plt.imshow(first_run, cmap='gray')
    plt.title('First Result: ' + os.path.basename(output_path))  # Display file name
    plt.axis('off')

    # Ground Truth Image
    plt.subplot(1, 4, 2)
    plt.imshow(second_run, cmap='gray')
    plt.title('Second Run: ' + os.path.basename(label_path))  # Display file name
    plt.axis('off')

    # Post-Processed Image
    plt.subplot(1, 4, 3)
    plt.imshow(original, cmap='gray')
    plt.title('Original: ' + os.path.basename(postprocessed_path))  # Display file name
    plt.axis('off')

    # Label Image
    plt.subplot(1, 4, 4)
    plt.imshow(label_image, cmap='gray')
    plt.title('Label: ' + os.path.basename(label_path))  # Display file name
    plt.axis('off')

    plt.show()
