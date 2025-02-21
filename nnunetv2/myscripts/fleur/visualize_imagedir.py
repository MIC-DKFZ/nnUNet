import os
import matplotlib.pyplot as plt
from skimage import io

# Directory containing the images
image_directory = r'C:\Users\Test\Desktop\Fleur\Boundary_polygon_Unet\images'

# Get the list of image files in the directory (adjust to your image format)
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

# Loop through all images and display them one by one
for i, image_file in enumerate(image_files):
    # Full path to the image
    image_path = os.path.join(image_directory, image_file)

    # Load the image
    image = io.imread(image_path)

    # Create a plot for the image
    plt.imshow(image, cmap='gray')  # Use 'gray' for grayscale images or remove cmap for color images
    plt.title(f"Image {i + 1}: {image_file}")
    plt.axis('off')  # Turn off the axis labels

    # Show the plot
    plt.show()

    # Wait for the user to close the image window and continue to the next image
    input("Press Enter to view the next image...")
