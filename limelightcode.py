import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

invert_angle = False
previous_angle = None

def reduce_noise(img):
    # convert image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
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

def runPipeline(image, llrobot):
    global invert_angle
    global previous_angle
    
    #constants for code
    lower_threshold = np.array([15, 0, 120])   # determined experimentally
    upper_threshold = np.array([31, 255, 255])   # determined experimentally

    #initialize variables in case they return nothing
    max_area_contour = np.array([[]])
    llpython = [0,0,0]

    noise_reduction = reduce_noise(image)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return max_area_contour,image,llpython

    max_area_contour = contours[0]
    for c in contours:
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1000 or 300000 < area:
            continue

        if area > cv.contourArea(max_area_contour):
            max_area_contour = c

    M = cv.moments(max_area_contour)

    # premature return if stuff is wrong
    if M['m00'] == 0 or len(max_area_contour) <= 5:
        return max_area_contour,image,llpython

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    llpython[1] = cX
    llpython[2] = cY
    # draw the contour and center of the shape on the image
    cv.circle(image, (cX, cY), 7, (0, 0, 0), -1)

    leftmost   = tuple(max_area_contour[max_area_contour[:,:,0].argmin()][0])
    rightmost  = tuple(max_area_contour[max_area_contour[:,:,0].argmax()][0])
    topmost    = tuple(max_area_contour[max_area_contour[:,:,1].argmin()][0])
    bottommost = tuple(max_area_contour[max_area_contour[:,:,1].argmax()][0])

    for point in [leftmost, rightmost, topmost, bottommost]:
        cv.circle(image, point, 5, (0, 0, 0), 2)

    _,_,angle = cv.fitEllipse(max_area_contour)

    count = sum(
        1 for point in [leftmost, rightmost, topmost, bottommost] \
        if point[1] > cY
        )

    if should_invert(angle, count):
        angle += 180

    image = cv.putText(image, str(round(angle)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)    
    llpython[0] = angle

    return max_area_contour,image,llpython
  