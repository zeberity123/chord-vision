import os
import time
import cv2
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture, resize_image

# Define the directory containing the images
image_dir = 'guitar-fingering-recognition-master\pictures'

# Get the list of image files
image_files = os.listdir(image_dir)

# Calculate the number of subplots required (2 subplots per image)
num_images = len(image_files)
num_subplots = num_images * 2

# Calculate the number of rows and columns for the subplots
num_rows = (num_subplots + 1) // 2
num_cols = 2

# Initialize the plot
plt.figure(figsize=(10, num_rows * 5))

# Process each image
for idx, filename in enumerate(image_files):
    try:
        print(f"File found: {filename} - Processing...")
        start_time = time.time()

        # Load and process the image
        chord_image = Image(path=os.path.join(image_dir, filename))
        resized_image = resize_image(chord_image.image)
        resized_image = Image(img=resized_image)
        rotated_image = rotate_neck_picture(resized_image)
        cropped_image = crop_neck_picture(rotated_image)

        # Display the original and processed images
        plt.subplot(num_rows, num_cols, 2 * idx + 1)
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original: {filename}")

        plt.subplot(num_rows, num_cols, 2 * idx + 2)
        plt.imshow(cv2.cvtColor(cropped_image.image, cv2.COLOR_BGR2RGB))
        plt.title(f"Processed: {filename}")

        print(f"Done - Time elapsed: {round(time.time() - start_time, 2)} seconds")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Show the plot
plt.tight_layout()
plt.show()
