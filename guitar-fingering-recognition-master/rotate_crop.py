import cv2
import numpy as np
from statistics import median
from math import inf
from image import Image
from functions import threshold, rotate

def rotate_neck_picture(image):
    """
    Rotates the picture so that the neck of the guitar is horizontal.
    Uses Hough transform to detect lines and calculates the median slope to determine the rotation angle.

    :param image: an Image object
    :return rotated_neck_picture: an Image object rotated according to the angle of the median slope detected in the input image
    """
    image_to_rotate = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    # Perform Hough Line Transform
    lines = image.lines_hough_transform(edges, 50, 50)  # TODO: Calibrate params automatically
    slopes = []

    # Calculate the slopes of detected lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                slope = abs((y2 - y1) / (x2 - x1))
                slopes.append(slope)
            except ZeroDivisionError:
                continue

    # Calculate median slope and the rotation angle
    if slopes:
        median_slope = median(slopes)
        angle = np.degrees(np.arctan(median_slope))
        rotated_image = rotate(image_to_rotate, -angle)
    else:
        rotated_image = image_to_rotate

    return Image(img=rotated_image)


def crop_neck_picture(image):
    """
    Crops the picture to focus on the region of interest (i.e., the guitar neck).
    Detects horizontal lines and crops around the densest region.

    :param image: an Image object of the neck (rotated horizontally if necessary)
    :return cropped_neck_picture: an Image object cropped around the neck
    """
    image_to_crop = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    # Perform Hough Line Transform
    lines = image.lines_hough_transform(edges, 50, 50)  # TODO: Calibrate params automatically
    y_coords = []

    # Collect all y-coordinates of the detected lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            y_coords.append(y1)
            y_coords.append(y2)

    # Sort y-coordinates and find the densest region
    if y_coords:
        y_sorted = sorted(y_coords)
        y_differences = [y_sorted[i + 1] - y_sorted[i] for i in range(len(y_sorted) - 1)]

        first_y = 0
        last_y = inf

        for i in range(len(y_differences)):
            if y_differences[i] == 0:
                last_y = y_sorted[i + 1]
                if first_y == 0:
                    first_y = y_sorted[i]

        cropped_image = image_to_crop[max(first_y - 10, 0):min(last_y + 10, image_to_crop.shape[0])]
    else:
        cropped_image = image_to_crop

    return Image(img=cropped_image)


def resize_image(img):
    """
    Recursively resizes the image if the resolution is too high.

    :param img: an image as defined in OpenCV
    :return resized_image: an image as defined in OpenCV with reduced resolution
    """
    height, width = img.shape[:2]
    if height >= 1080 or width >= 1920:
        resized_image = cv2.resize(img, (int(width * 0.8), int(height * 0.8)))
        return resize_image(resized_image)
    else:
        return img


if __name__ == "__main__":
    print("Run rotate_crop_tests.py to have a look at results!")
