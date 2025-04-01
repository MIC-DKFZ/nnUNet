import json
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the .json files
directory = r'C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset005\Seg25\Oct 16 2024 15_03'
output_directory = r'C:\Users\Test\Desktop\Fleur\3_Seg\nnUNet_raw\Dataset005\output'

# Extract segmentation identifier from the directory name
segmentation_identifier = os.path.basename(os.path.dirname(directory))

# Define a mapping for class IDs to consecutive values
label_mapping = {
    0: 0,  # background
    1: 1,  # boundary
    2: 2,  # segment to remove
    3: 3,  # healthy
}

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Counter for naming images (starting from 0)
image_counter = 0

# Process each .json file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        json_path = os.path.join(directory, filename)
        
        # Load JSON data
        with open(json_path) as f:
            data = json.load(f)

        # Check if data is a list or a dictionary
        if isinstance(data, list):
            data = data[0]  # Adjust based on your actual data structure
        
        # Count occurrences of each classId
        class_count = {}
        for instance in data.get('instances', []):
            class_id = instance['classId']
            class_count[class_id] = class_count.get(class_id, 0) + 1

        # Print the total count of each class
        print(f'Counts for {filename}:')
        for class_id, count in class_count.items():
            print(f'Class ID {class_id}: {count} occurrences')

        # Create a blank image (1280x1024) with mode 'L' for grayscale
        width, height = 1280, 1024
        image = Image.new('L', (width, height), 0)  # black background
        draw = ImageDraw.Draw(image)

        # Draw polygons and polylines
        for instance in data.get('instances', []):
            points = [(x, y) for x, y in zip(instance['points'][::2], instance['points'][1::2])]
            if instance['type'] == 'polygon':
                # Use the mapped value for filling
                mapped_value = label_mapping.get(instance['classId'], 0)  # Default to 0 (background)
                draw.polygon(points, outline=mapped_value, fill=mapped_value)
            elif instance['type'] == 'polyline':
                # Draw polyline with mapped class ID value
                draw.line(points, fill=label_mapping.get(1, 0), width=2)  # Default to classId 1 (boundary)

        # Create a new name for the output image, including the segmentation identifier
        image_name = f'{segmentation_identifier}_image_{image_counter:03d}.png'  # Format as identifier_image_000, etc.
        image_counter += 1  # Increment the counter

        # Save the image in the specified output directory
        image.save(os.path.join(output_directory, image_name))
        print(f'Saved image: {image_name} in {output_directory}')

        # Convert image to numpy array to check unique pixel values
        image_np = np.array(image)
        unique_values = np.unique(image_np)
        print(f'Unique pixel values in {image_name}: {unique_values}') 

        # Check if the image is completely black
        if np.all(image_np == 0):
            print(f'Warning: {image_name} is completely black!')
        else:
            print(f'{image_name} contains non-black pixels.')

        # Visualize the image using matplotlib
        plt.imshow(image_np, cmap='gray')
        plt.title(image_name)
        plt.axis('off')  # Hide axes
        plt.show()  # Display the image
