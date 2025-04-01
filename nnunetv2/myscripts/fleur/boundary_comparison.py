import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color

# Pad naar je beelden
image_path_1 = r'C:\Users\Test\Desktop\Fleur\seg25.png'
image_path_2 = r'C:\Users\Test\Desktop\Fleur\Boundary_EdgeDetection\image_010.png'

# Laad het eerste beeld (seg4.png) en converteer naar grayscale
image_1 = io.imread(image_path_1)
if image_1.shape[-1] == 4:  # RGBA (4 channels)
    image_1 = image_1[:, :, :3]
grayscale_image_1 = color.rgb2gray(image_1)

# Laad het tweede beeld (image_001.png) en converteer naar grayscale
image_2 = io.imread(image_path_2)
if image_2.shape[-1] == 4:  # RGBA (4 channels)
    image_2 = image_2[:, :, :3]
grayscale_image_2 = color.rgb2gray(image_2)

# Print dimensions of both images
print(f"seg4 image shape: {image_1.shape}")
print(f"Template image shape: {image_2.shape}")

# Resize the template to fit within seg4
height_ratio = image_1.shape[0] / image_2.shape[0]
width_ratio = image_1.shape[1] / image_2.shape[1]
resize_ratio = min(height_ratio, width_ratio)

# Resize template image to fit inside seg4 image
resized_template = cv2.resize(image_2, (0, 0), fx=resize_ratio, fy=resize_ratio)

# Print resized template size
print(f"Resized template size: {resized_template.shape}")

# Converteer naar 8-bit beelden (voor gebruik met OpenCV)
grayscale_image_1 = np.uint8(grayscale_image_1 * 255)  # Normaliseren naar 0-255
resized_template_gray = np.uint8(color.rgb2gray(resized_template) * 255)  # Normalized resized template

# Step 2: Template matching with OpenCV (grayscale)
result = cv2.matchTemplate(grayscale_image_1, resized_template_gray, cv2.TM_CCOEFF_NORMED)

# Get the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Get the top-left corner of the best match location
top_left = max_loc
h, w, _ = resized_template.shape  # Height and width of the resized template, ignoring channels

# Output the best match location and the dimensions of the images
print(f"Best match location (top-left): {top_left}")
print(f"Template size (width x height): {w} x {h}")

# Check if the match coordinates are within the image bounds
image_1_height, image_1_width = image_1.shape[:2]
print(f"seg4 image bounds: Width: {image_1_width}, Height: {image_1_height}")

# Ensure that the match coordinates are within the bounds of seg4 image
if top_left[0] < 0 or top_left[1] < 0 or top_left[0] + w > image_1_width or top_left[1] + h > image_1_height:
    print("Warning: Match coordinates are out of bounds!")
else:
    print("Match coordinates are within bounds.")

# Visualize the images and the result
plt.figure(figsize=(12, 8))

# Show the original seg4 image
plt.subplot(2, 2, 1)
plt.imshow(image_1)
plt.title("Original Image (seg4.png)")
plt.axis('on')  # Keep axis on to visualize coordinates

# Show the second image (resized template)
plt.subplot(2, 2, 2)
plt.imshow(resized_template)
plt.title("Resized Template Image")
plt.axis('off')

# Show the seg4 image with the hollow circle indicating the best match location
plt.subplot(2, 2, 3)
image_with_circle = np.copy(image_1)
center_coordinates = (top_left[0] + w // 2, top_left[1] + h // 2)
cv2.circle(image_with_circle, center_coordinates, 10, (255, 0, 0), 2)  # Hollow red circle with thickness=2
plt.imshow(image_with_circle)
plt.title("seg4.png with Best Match")
plt.axis('on')

# Remove the heatmap from visualization (we don't need this anymore)

plt.tight_layout()
plt.show()

# Output the best match score and location
print(f"Best match score (correlation coefficient): {max_val}")
