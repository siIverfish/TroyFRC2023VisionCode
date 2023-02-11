import cv2 as cv
import numpy as np


def reduce_noise(img, lower_threshold, upper_threshold):
    # convert image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    cv.imshow("noise_reduction", noise_reduction)
    return noise_reduction


previous_angle = None
invert_angle = False

def should_invert(angle, count):
    global previous_angle, invert_angle

    if not previous_angle:
        previous_angle = angle
        return False

    if angle - previous_angle > 90: # jumping from 0 to 180 degrees
        if count > 2: # cone tip pointing down
            return True
        else: # cone tip pointing ups
            return False
    elif angle - previous_angle < -90: # jumping from 180 to 0 degrees
        if count > 2: # cone tip pointing down
            return False
        else: # cone tip pointing up
            return True


def maximum_contour(contours):
    return max(
        (c for c in contours if 1000 < c < 30_000),
        key=cv.contourArea
    )


def maximum_contour_center(contours):
    M = cv.moments(maximum_contour(contours))

    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    return np.array(center_x, center_y)

