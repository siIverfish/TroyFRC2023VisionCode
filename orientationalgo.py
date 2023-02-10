import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import math



def reduce_noise(img):
    # convert image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    return noise_reduction


def get_maximum_contour(contours):
    return max(
        contour for contour in contours \
        if cv.contourArea(contour) > 1000 and cv.contourArea(contour) < 300000
    )


def runPipeline(image, llrobot):
    #constants for code
    lower_threshold = np.array([15, 0, 120])   # determined experimentally
    upper_threshold = np.array([31, 255, 255])   # determined experimentally

    #initialize variables in case they return nothing
    max_area_contour = np.array([[]])
    llpython = [0,0,0]

    # convert image to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    noise_reduction = reduce_noise(hsv)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return max_area_contour,image,llpython

    max_area_contour = get_maximum_contour(contours)

    M = cv.moments(max_area_contour)

    # stop if things are bad
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

    # nice circles
    for point in [leftmost, rightmost, topmost, bottommost]:
        cv.circle(image, point, 5, (0, 0, 0), 2)

    # the first two returns are not used
    _, _, angle = cv.fitEllipse(max_area_contour)

    slope = math.atan(angle + 90) # I have no idea what I'm doing

    # the number of points where the condition is true
    # about 15 fewer lines of code than the alternative
    count_top = sum(
        1 for point in max_area_contour \
        if point[0] * slope < point[1]
        )
    
    # the bottom is the opposite of the top
    count_bottom = len(max_area_contour) - count_top

    print(count_top, count_bottom)

    image = cv.putText(image, str(round(angle)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)   

    llpython[0] = angle

    return max_area_contour,image,llpython
  