import os
import time
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from grid_detection import string_detection, fret_detection
import cv2

def process_images(detection_function):
    """
    General function to process images and apply a given detection function.
    
    :param detection_function: A function that applies a specific detection algorithm to the image
    """
    image_dir = 'pictures'
    image_files = os.listdir(image_dir)
    num_images = len(image_files)
    num_subplots = num_images * 2
    num_rows = (num_subplots + 1) // 2
    num_cols = 2

    plt.figure(figsize=(10, num_rows * 5))

    for idx, filename in enumerate(image_files):
        try:
            print(f"File found: {filename} - Processing...")
            start_time = time.time()
            chord_image = Image(path=image_dir, img=filename)
            rotated_image = rotate_neck_picture(chord_image)
            cropped_image = crop_neck_picture(rotated_image)
            detection_result = detection_function(cropped_image)

            plt.subplot(num_rows, num_cols, 2 * idx + 1)
            plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
            plt.title(f"Original: {filename}")

            plt.subplot(num_rows, num_cols, 2 * idx + 2)
            plt.imshow(cv2.cvtColor(detection_result.image, cv2.COLOR_BGR2RGB))
            plt.title(f"Processed: {filename}")

            print(f"Done - Time elapsed: {round(time.time() - start_time, 2)} seconds")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    plt.tight_layout()
    plt.show()

def string_detection_tests():
    """
    Runs tests for string detection on images in the specified directory.
    """
    process_images(lambda img: string_detection(img)[1])

def fret_detection_tests():
    """
    Runs tests for fret detection on images in the specified directory.
    """
    process_images(fret_detection)

def grid_detection_tests():
    """
    Runs tests for detecting both strings and frets on images in the specified directory.
    """
    def detect_grid(cropped_image):
        neck_strings = string_detection(cropped_image)[0]
        neck_fret = fret_detection(cropped_image)
        for string, pts in neck_strings.separating_lines.items():
            cv2.line(neck_fret.image, pts[0], pts[1], (127, 0, 255), 2)
        return neck_fret

    process_images(detect_grid)

if __name__ == "__main__":
    print("What would you like to detect? \n\t1 - Strings \n\t2 - Frets \n\t3 - Strings and frets")
    choice = input("[1/2/3] > ")
    if choice == "1":
        print("Detecting strings...")
        string_detection_tests()
    elif choice == "2":
        print("Detecting frets...")
        fret_detection_tests()
    elif choice == "3":
        print("Detecting whole grid...")
        grid_detection_tests()
    else:
        print("Command not defined - Aborted.")
