import cv2
import numpy as np
# import time

cube = False 
cap = cv2.VideoCapture(1+cv2.CAP_DSHOW)

x_res = 720 #determine later
y_res = 480 #determine later

center_coord = np.array([x_res/2, y_res/2])


def reduce_noise(img, lower_threshold, upper_threshold):
    # convert image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    return noise_reduction


def maximum_contour_center(contours):
    center = np.zeros(2)
    for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            if M["m00"] == 0:
                continue
            
            # calculate area of contour
            if(M['m00'] > max_area):
                max_area = M['m00']
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                center[0] = center_x
                center[1] = center_y
    return center

# kinda long name 
def escape_if_user_exits():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

def process_cube():
    lower_threshold = np.array([118,87,86]) 
    upper_threshold = np.array([133,255,255])
    while True:
        _, frame = cap.read()
        noise_reduction = reduce_noise(frame, lower_threshold, upper_threshold)

        contours, _ = cv2.findContours(noise_reduction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        center = maximum_contour_center(contours)
        
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)

        cv2.imshow('frame', frame)
        cv2.imshow('noise_reduction', noise_reduction)

        escape_if_user_exits()

# I think this is for cones
def process_cone():
    lower_threshold = np.array([18, 44, 101])   # determined experimentally
    upper_threshold = np.array([31, 232, 255])   # determined experimentally
    while True:
        # getting video frame
        _, frame = cap.read()
        noise_reduction = reduce_noise(frame, lower_threshold, upper_threshold)
        
        # calculate x,y coordinate of center
        contours,_ = cv2.findContours(noise_reduction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        center = maximum_contour_center(contours)
        
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)
        cv2.imshow("result",noise_reduction)
        cv2.imshow("normal",frame)

        escape_if_user_exits()

if cube:
    process_cube()
else:
    process_cone()