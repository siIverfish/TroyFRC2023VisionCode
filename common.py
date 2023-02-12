import cv2 as cv
import numpy as np


def infinite_frame_stream():
    """
    Returns a generator that yields frames from the webcam.
    Also exits the program if the user presses 'q' and waits between frames.
    """
    # cap = cv.VideoCapture(1 + cv.CAP_DSHOW)
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if frame is None:
            print("Error reading frame")
            exit(1)
        yield frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            exit(0)

def reduce_noise(img, lower_threshold, upper_threshold):
    """
    This function does most of the heavy lifting for object detection.
    It returns a binary image with the object of interest (between lower & upper threshold) in 
    white and the rest in black.
    """
    # convert image to HSV
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    # thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    img = cv.inRange(img, lower_threshold, upper_threshold)

    img = cv.erode(img, np.ones((10, 10), np.uint8), iterations = 1)
    img = cv.blur(img,(15,15))
    img = cv.inRange(img, 169, 255)
    
    return img


previous_angle = None
invert_angle = False

def should_invert(angle, count):
    """
    Inverts the angle sometimes?
    """
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


def get_maximum_contour(contours):
    """
    Returns the contour with the largest area in the list of contours.
    """
    contours = [
        contour for contour in contours \
        if 1_000 < cv.contourArea(contour) < 300_000
    ]
    if len(contours) == 0:
        return None
    return max(
        contours,
        key=cv.contourArea
    )

def get_maximum_contour_center(contours):
    """
    Returns the center of the contour with the largest area in the list of contours.
    """
    max_contour = maximum_contour(contours)
    if max_contour is None:
        return None
    M = cv.moments(max_contour)

    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    return np.array([center_x, center_y])


# kinda long name 
def escape_if_user_exits():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)


