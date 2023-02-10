import cv2 as cv
from math import atan2, cos, sin, sqrt, pi, pow
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

cap = cv.VideoCapture(0)
 
lower_threshold = np.array([15, 120, 120])   # determined experimentally
upper_threshold = np.array([31, 255, 255])   # determined experimentally

invert_angle = False
previous_angle = None

def findDistance(x1, y1, x2, y2):
    return pow(pow((x1 - x2), 2) + pow((y1 - y2), 2), 1/2)

def load_image():
    # Load the image
    _, img = cap.read()

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)
    return img

def reduce_noise(img):
    # convert image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    return noise_reduction

def output(image):
    cv.imshow('Output Image', image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        exit(0)

def get_maximum_contour(contours):
    max_area_contour = contours[0]
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1000 or 300000 < area:
            continue

        if area > cv.contourArea(max_area_contour):
            max_area_contour = c
    return max_area_contour

while True:
    img = load_image()
    noise_reduction = reduce_noise(img)
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # stop if no contours are found
    if len(contours) == 0:
        output(img)
        continue

    max_area_contour = get_maximum_contour(contours)

    M = cv.moments(max_area_contour)

    # stop if the first moment is 0 ?
    if M['m00'] == 0:
        output(img)
        continue

    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    # draw the contour and center of the shape on the image
    cv.circle(img, (center_x, center_y), 7, (0, 0, 0), -1)

    # someone make a comment to explain this
    leftmost   = tuple(max_area_contour[max_area_contour[:,:,0].argmin()][0])
    rightmost  = tuple(max_area_contour[max_area_contour[:,:,0].argmax()][0])
    topmost    = tuple(max_area_contour[max_area_contour[:,:,1].argmin()][0])
    bottommost = tuple(max_area_contour[max_area_contour[:,:,1].argmax()][0])

    # draws nice circles for the nice output image
    for point in [leftmost, rightmost, topmost, bottommost]:
        cv.circle(img, point, 5, (0, 0, 0), 2)

    (x,y),(MA,ma),angle = cv.fitEllipse(max_area_contour)

    largestDistance = -1
    for point in [leftmost, rightmost, topmost, bottommost]:
        # if the distance is the new maximum
        if findDistance(point[0], point[1], center_x, center_y) > largestDistance:
            # set the new maximum
            largestDistance = findDistance(point[0], point[1], center_x, center_y)
            tipOfCone = point
    
    # if the top of the cone is to the right of the center ?
    if tipOfCone[0] > center_x:
        invert_angle = True
    else:
        invert_angle = False

    # inverts the angle
    if invert_angle:
        angle += 180
    
    print(angle)
    output(img)
  
# Save the output image to the current directory
# cv.imwrite("output_img.jpg", img)