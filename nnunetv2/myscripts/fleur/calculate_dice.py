import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_dice_coefficient(image1, image2):
    # Ensure the images are binary (0s and 1s)
    binary_image1 = (image1 > 0).astype(np.uint8)
    binary_image2 = (image2 > 0).astype(np.uint8)

    # Calculate the intersection and sums
    intersection = np.sum(binary_image1 * binary_image2)
    dice_coefficient = (2. * intersection) / (np.sum(binary_image1) + np.sum(binary_image2))
    
    return dice_coefficient

# Define the directories
label_directory = r'C:\Users\Test\Desktop\Fleur\Run2\Validatie_set\output_labels'
result_directory = r'C:\Users\Test\Desktop\Fleur\3_Seg\results_data_augmentation'

# Initialize variables for mean Dice calculation
total_dice = 0.0
num_images = 18  # Number of images

# Loop through image numbers 1 to 18
for i in range(1, num_images + 1):
    # Format the image number with leading zeros
    image_number = f'image_{i:03d}.png'
    
    # Load the images
    image_path1 = os.path.join(label_directory, image_number)
    image_path2 = os.path.join(result_directory, image_number)
    
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    # Calculate Dice coefficient
    dice_score = calculate_dice_coefficient(image1, image2)
    total_dice += dice_score
    
    print(f'Dice Coefficient for {image_number}: {dice_score:.4f}')

    # Display the images side by side with the Dice score
    plt.figure(figsize=(12, 6))

    # Label Image
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(f'Label: {image_number}')  # Display file name
    plt.axis('off')

    # Result Image
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(f'Result: {image_number} (Dice: {dice_score:.4f})')  # Display Dice score
    plt.axis('off')

    plt.show()

# Calculate and print the mean Dice coefficient
mean_dice = total_dice / num_images
print(f'Mean Dice Coefficient: {mean_dice:.4f}')

