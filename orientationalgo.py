import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import math

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
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if not len(contours) == 0:
        max_area_contour = contours[0]
        for c in contours:
            area = cv.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1000 or 300000 < area:
                continue

            if area > cv.contourArea(max_area_contour):
                max_area_contour = c

        M = cv.moments(max_area_contour)
        if not M['m00'] == 0 and len(max_area_contour) > 5:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            llpython[1] = cX
            llpython[2] = cY
            # draw the contour and center of the shape on the image
            cv.circle(image, (cX, cY), 7, (0, 0, 0), -1)

            leftmost = tuple(max_area_contour[max_area_contour[:,:,0].argmin()][0])
            rightmost = tuple(max_area_contour[max_area_contour[:,:,0].argmax()][0])
            topmost = tuple(max_area_contour[max_area_contour[:,:,1].argmin()][0])
            bottommost = tuple(max_area_contour[max_area_contour[:,:,1].argmax()][0])

            cv.circle(image, leftmost, 5, (0, 0, 0), 2)
            cv.circle(image, rightmost, 5, (0, 0, 0), 2)
            cv.circle(image, topmost, 5, (0, 0, 0), 2)
            cv.circle(image, bottommost, 5, (0, 0, 0), 2)

            (x,y),(MA,ma),angle = cv.fitEllipse(max_area_contour)

            slope = math.atan(angle + 90) # I have no idea what I'm doing
            countTop,countBottom = 0,0
            if leftmost[0] * slope >= leftmost[1]:
                countBottom += 1
            else:
                countTop += 1
            if rightmost[0] * slope >= rightmost[1]:
                countBottom += 1
            else:
                countTop += 1
            if topmost[0] * slope >= topmost[1]:
                countBottom += 1
            else:
                countTop += 1
            if bottommost[0] * slope >= bottommost[1]:
                countBottom += 1
            else:
                countTop += 1

            print(countTop, countBottom)

            image = cv.putText(image, str(round(angle)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)    
            llpython[0] = angle

    return max_area_contour,image,llpython
  