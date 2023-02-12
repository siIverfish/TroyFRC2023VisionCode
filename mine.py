import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

from common import infinite_frame_stream, maximum_contour_center



class Cone:
    """ The lower and upper bounds for a printed image of the cone. """
    # LOWER = np.array([HSV_COLOR[i] - DIFFERENCES[i] for i in range(3)])
    # UPPER = np.array([HSV_COLOR[i] + DIFFERENCES[i] for i in range(3)])
    LOWER = np.array([8, 44, 101])
    UPPER = np.array([21, 232, 255])


for frame in infinite_frame_stream():
    frame = cv.medianBlur(frame, 21)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # cv.imshow('color', frame)

    frame = cv.inRange(frame, Cone.LOWER, Cone.UPPER)

    # cv.imshow('binary', frame)
    
    contours, _ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    

