import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
 
DELTAVALUE = 20 # variation for how wide the angle "upright" can be, in each direction
 
def returnOrientation(angle): # return true if angle is within upright range
    print(angle)
    if angle < (90 + DELTAVALUE) and angle > (90 - DELTAVALUE):
        return True
        # return True
    else:
        return False
        # return False

cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
 
lower_threshold = np.array([18, 44, 101])   # determined experimentally
upper_threshold = np.array([31, 232, 255])   # determined experimentally

invert_angle = False
previous_angle = None

while True:
    # Load the image
    ret, img = cap.read()

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)

    cv.imshow('Input Image', img)

    # convert image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.blur(thresh,(20,20))
    noise_reduction = cv.inRange(noise_reduction, 1, 75)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if not len(contours) == 0:
        max_area_contour = contours[0]

    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        if area > cv.contourArea(max_area_contour):
            max_area_contour = c
        
        # Ignore contours that are too small or too large
        if area < 1000 or 300000 < area:
            continue
        
        #india

        M = cv.moments(max_area_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv.circle(img, (cX, cY), 7, (0, 0, 0), -1)

        leftmost = tuple(max_area_contour[max_area_contour[:,:,0].argmin()][0])
        rightmost = tuple(max_area_contour[max_area_contour[:,:,0].argmax()][0])
        topmost = tuple(max_area_contour[max_area_contour[:,:,1].argmin()][0])
        bottommost = tuple(max_area_contour[max_area_contour[:,:,1].argmax()][0])

        cv.circle(img, leftmost, 5, (0, 0, 0), 2)
        cv.circle(img, rightmost, 5, (0, 0, 0), 2)
        cv.circle(img, topmost, 5, (0, 0, 0), 2)
        cv.circle(img, bottommost, 5, (0, 0, 0), 2)

        # Draw each contour only for visualisation purposes
        cv.drawContours(img, contours, i, (0, 0, 255), 2)

        (x,y),(MA,ma),angle = cv.fitEllipse(max_area_contour)
        
        # ave_y = (leftmost[1] + rightmost[1] + topmost[1] + bottommost[1]) / 4

        # if previous_angle is None or abs(angle - previous_angle) > 90:
        #     if ave_y > cY:
        #         invert_angle = True
        #     else:
        #         invert_angle = False

        count = 0
        if leftmost[1] > cY:
          count += 1
        if rightmost[1] > cY:
          count += 1
        if topmost[1] > cY:
          count += 1
        if bottommost[1] > cY:
          count += 1
        
        if previous_angle is None or angle - previous_angle > 90: # jumping from 0 to 180 degrees
            if count > 2: # cone tip pointing down
                invert_angle = False
            else: # cone tip pointing up
                invert_angle = True
        elif angle - previous_angle < -90: # jumping from 180 to 0 degrees
            if count > 2: # cone tip pointing down
                invert_angle = True
            else: # cone tip pointing up
                invert_angle = False

        previous_angle = angle
        
        if invert_angle:
            print (angle)
        else:
            print (angle + 180)
        
        # Find the orientation of each shape
        #angle = getOrientation(max_area_contour, img)
        #print (returnOrientation(np.rad2deg(angle))) # function to print out if upright or not

    cv.imshow('Output Image', img)
    cv.imshow('Noise Reduction', noise_reduction)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# Save the output image to the current directory
# cv.imwrite("output_img.jpg", img)