import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology

# Define the folder containing the images
prediction_folder = r'C:\Users\Test\Desktop\Fleur\extract_boundaries\Best_prediction'
output_folder = r'C:\Users\Test\Desktop\Fleur\Boundary_EdgeDetection'  # Folder to save processed images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the folder
for filename in os.listdir(prediction_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Check for image files
        # Load the image
        image_path = os.path.join(prediction_folder, filename)
        image = io.imread(image_path)

        # Convert to grayscale if the image is in RGB
        if image.ndim == 3:
            gray_image = np.mean(image, axis=2)  # Average the RGB channels
        else:
            gray_image = image

        # Normalize the gray image
        normalized_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image) + 1e-8)

        # Apply Sobel edge detection
        edges = filters.sobel(normalized_image)

        # Threshold to create a binary mask of edges
        thresholded_edges = edges > 0.1  # Adjust threshold as necessary

        # Create a mask for gray structures (non-black regions)
        gray_mask = (normalized_image > 0.2) & (normalized_image < 0.8)  # Adjust based on your image

        # Create a mask for black pixels
        black_mask = normalized_image < 0.1  # Adjust based on your image

        # Combine the edge mask with the gray mask
        combined_mask = thresholded_edges & gray_mask

        # Remove edges around black pixels within gray structures
        combined_mask = combined_mask & ~black_mask  # Exclude black pixels from the edges

        # Dilate the edges to make them thicker
        dilated_edges = morphology.dilation(combined_mask, morphology.disk(3))  # Change the disk size for thickness

        # Create an RGB overlay for the edges
        edge_overlay = np.zeros((*gray_image.shape, 3), dtype=np.float32)
        edge_overlay[dilated_edges] = [1, 0, 0]  # Red color for the edges

        # Create the combined image
        combined_image = np.zeros((*gray_image.shape, 3), dtype=np.float32)
        combined_image[..., 0] = normalized_image  # Red channel
        combined_image[..., 1] = normalized_image  # Green channel
        combined_image[..., 2] = normalized_image  # Blue channel
        combined_image[dilated_edges] = edge_overlay[dilated_edges]  # Add edges

        # Save the combined image
        output_path = os.path.join(output_folder, filename)
        io.imsave(output_path, (combined_image * 255).astype(np.uint8))  # Save as an image

        # Display the results
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image: {filename}')
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Image with Boundaries')
        plt.imshow(combined_image)
        plt.axis('off')

        plt.show()
