import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label

# Paths to the folders
ground_truth_folder = r'C:\Users\Test\Desktop\Fleur\extract_boundaries\GroundTruth_labels'
prediction_folder = r'C:\Users\Test\Desktop\Fleur\extract_boundaries\Best_prediction'
result_folder = r'C:\Users\Test\Desktop\Fleur\extract_boundaries\results_boundaryextraction'

# Create the result folder if it doesn't exist
os.makedirs(result_folder, exist_ok=True)

# Loop through all images in the ground truth folder
for filename in os.listdir(ground_truth_folder):
    if filename.endswith('.png'):  # Adjust if necessary for other file extensions
        # Load the ground truth and prediction
        ground_truth = plt.imread(os.path.join(ground_truth_folder, filename))
        prediction = plt.imread(os.path.join(prediction_folder, filename))

        # Ensure the images are in float format for processing
        ground_truth = ground_truth.astype(np.float32)
        prediction = prediction.astype(np.float32)

        # If the images are grayscale, convert them to RGB
        if ground_truth.ndim == 2:
            ground_truth = np.stack([ground_truth] * 3, axis=-1)  # Convert to RGB
        if prediction.ndim == 2:
            prediction = np.stack([prediction] * 3, axis=-1)  # Convert to RGB

        # Normalize images to range [0, 1]
        ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))
        prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))

        # Calculate the boundary as ground truth - prediction
        boundary = np.abs(ground_truth - prediction) > 0  # True where there is a difference

        # Create a color overlay for the boundaries
        boundary_color = np.array([1.0, 0.5, 0.5])  # Light red

        # Initialize result image
        result = np.zeros_like(prediction)

        # Assign the boundary color to the appropriate locations in the result
        for i in range(3):  # For each channel (R, G, B)
            result[..., i] = np.where(boundary[..., i], boundary_color[i], 0)  # Set boundary color or black

        # Combine the prediction image with the boundary overlay
        combined_result = prediction.copy()  # Start with the prediction image
        combined_result[boundary] = result[boundary]  # Apply boundary color where there is a boundary

        # Identify black pixels
        black_mask = np.all(prediction < 0.01, axis=-1)  # Identify black pixels

        # Label connected components of black pixels
        labeled_mask, num_features = label(black_mask)

        # Iterate through connected components
        for label_id in range(1, num_features + 1):
            # Create a mask for the current connected component
            component_mask = (labeled_mask == label_id)

            # Check if this component touches any boundary pixels
            if np.any(boundary[component_mask]):
                # If connected to a boundary, remove the boundary for this component
                combined_result[component_mask] = prediction[component_mask]  # Use prediction where the component is connected

        # Visualize the results using Matplotlib
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Ground Truth')
        plt.imshow(ground_truth)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Prediction')
        plt.imshow(prediction)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Result with Boundary')
        plt.imshow(combined_result)
        plt.axis('off')

        plt.show()

print("Boundary extraction and visualization completed.")
