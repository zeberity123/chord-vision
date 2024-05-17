import cv2
import numpy as np
from collections import defaultdict

def skin_detection(img):
    """
    Naively detects skin in the image. Non-skin pixels will be set to black (0, 0, 0).
    
    :param img: An image as defined in OpenCV.
    :return: An image as defined in OpenCV with non-skin areas set to black.
    """
    for index_line, line in enumerate(img):
        for index_pixel, pixel in enumerate(line):
            # Condition for skin detection (adjust thresholds as needed)
            if pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20 and max(pixel) - min(pixel) > 15 \
                    and abs(pixel[2] - pixel[1]) > 15 and pixel[2] > pixel[0] and pixel[2] > pixel[1] \
                    and index_pixel > len(line) / 2:
                pass
            else:
                img[index_line][index_pixel] = (0, 0, 0)

    return img

def locate_hand_region(img):
    """
    Refines hand region after skin detection by returning the region with the highest density
    of non-black pixels when looking at regions split vertically.
    
    :param img: An image as defined in OpenCV, after skin detection.
    :return: An image as defined in OpenCV with the detected hand region highlighted.
    """
    height, width = img.shape[:2]
    hand_region = np.zeros((height, width, 3), np.uint8)

    x_dict = defaultdict(int)
    for line in img:
        for j, pixel in enumerate(line):
            if pixel.any() > 0:
                x_dict[j] += 1

    if not x_dict:
        return hand_region

    max_density = max(x_dict.values())
    max_x_density = max(x_dict.keys(), key=(lambda k: x_dict[k]))
    min_x, max_x = min(x_dict.keys()), max(x_dict.keys())

    m, n = 0, 0
    last_density = x_dict[max_x_density]

    while max_x_density - m > min_x and x_dict[max_x_density - m] >= 0.1 * max_density and x_dict[max_x_density - m] >= 0.5 * last_density:
        m += 1
        last_density = x_dict[max_x_density - m]

    last_density = x_dict[max_x_density]
    while max_x_density + n < max_x and x_dict[max_x_density + n] >= 0.1 * max_density and x_dict[max_x_density + n] >= 0.5 * last_density:
        n += 1
        last_density = x_dict[max_x_density + n]

    tolerance = 20
    min_limit = max_x_density - m - tolerance
    max_limit = max_x_density + n + tolerance

    for i, line in enumerate(img):
        for j, pixel in enumerate(line):
            if min_limit < j < max_limit:
                hand_region[i][j] = img[i][j]

    return hand_region

def hand_detection(img):
    """
    Detects contours in the hand using Canny edge detection and tries to find fingertips.
    
    :param img: An image as defined in OpenCV, after skin detection and refining.
    :return: An image as defined in OpenCV with hand contours highlighted.
    """
    # Apply skin detection and locate hand region
    skin_detected = skin_detection(img)
    hand_region = locate_hand_region(skin_detected)
    
    # Blur the image and convert to grayscale for edge detection
    blurred = cv2.medianBlur(hand_region, 5)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 70, 100)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 5,
                               param1=100, param2=20, minRadius=20, maxRadius=90)
    
    # Draw circles if detected
    if circles is not None:
        circles = np.uint16(np.around(circles[0])) if len(circles) > 0 else None  # Check if circles contains any circles
        if circles is not None:
            for i in circles:
                # Draw the outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    return edges, img

if __name__ == "__main__":
    print("Run finger_detection_tests.py to have a look at results!")
