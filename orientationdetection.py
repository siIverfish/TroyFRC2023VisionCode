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

cap = cv.VideoCapture(1 + cv.CAP_DSHOW)
 
lower_threshold = np.array([18, 44, 170])   # determined experimentally
upper_threshold = np.array([31, 255, 255])   # determined experimentally

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
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if not len(contours) == 0:
        max_area_contour = contours[0]
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1000 or 300000 < area:
                continue

            if area > cv.contourArea(max_area_contour):
                max_area_contour = c

        #if not (cv.contourArea(max_area_contour) < 2000): #if there was no larger contour, must check the initial one
            M = cv.moments(max_area_contour)
            if not M['m00'] == 0:
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
                        invert_angle = True
                    else: # cone tip pointing up
                        invert_angle = False
                elif angle - previous_angle < -90: # jumping from 180 to 0 degrees
                    if count > 2: # cone tip pointing down
                        invert_angle = False
                    else: # cone tip pointing up
                        invert_angle = True

                previous_angle = angle

                if invert_angle:
                    angle += 180
                print (angle)

    cv.imshow('Output Image', img)
    cv.imshow('Noise Reduction', noise_reduction)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# Save the output image to the current directory
# cv.imwrite("output_img.jpg", img)