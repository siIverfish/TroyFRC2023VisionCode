import cv2 as cv
import numpy as np

from common import reduce_noise, maximum_contour_center

cube = False

def infinite_frame_stream():
    cap = cv.VideoCapture(1+cv.CAP_DSHOW)
    while True:
        _, frame = cap.read()
        if (cv.waitKey(1) & 0xFF == ord('q')) or frame is None:
            exit(0)
        yield frame

def make_circle(img, position, radius, color):
    cv.circle(img, (int(position[0]), int(position[1])), radius, color, -1)

def get_contours(img):
    contours, _ = cv.findContours(
        img, 
        cv.RETR_TREE, 
        cv.CHAIN_APPROX_SIMPLE
    )
    return contours


def process_cube():
    lower_threshold = np.array([118, 87, 86])
    upper_threshold = np.array([133, 255, 255])
    for frame in infinite_frame_stream():
        noise_reduction = reduce_noise(frame, lower_threshold, upper_threshold)

        contours = get_contours(noise_reduction)
        center = maximum_contour_center(contours)
        
        make_circle(frame, center, 5, (255, 255, 255))

        cv.imshow('frame', frame)
        cv.imshow('noise_reduction', noise_reduction)


def process_cone():
    lower_threshold = np.array([18, 44, 101])   # determined experimentally
    upper_threshold = np.array([31, 232, 255])   # determined experimentally
    for frame in infinite_frame_stream():
        cv.imshow("normal", frame)
        
        noise_reduction = reduce_noise(frame, lower_threshold, upper_threshold)
        
        # cv.imshow("result", noise_reduction)

        contours = get_contours(noise_reduction)
        if len(contours) == 0:
            print("no contours found")
            continue
        center = maximum_contour_center(contours)

        make_circle(frame, center, 5, (255, 255, 255))
        
        # cv.imshow("normal", frame)


if cube:
    process_cube()
else:
    process_cone()
