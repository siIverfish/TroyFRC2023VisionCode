import cv2 as cv
import numpy as np

from common import (
    reduce_noise,
    maximum_contour_center,
    infinite_frame_stream,
)

CUBE = False

class Threshold:
    """
    Holds the lower and upper threshold color values for the objects.
    If the RGB values of a pixel are within these bounds, it will be
    considered part of the object.
    """

    class Cone:
        "The lower and upper bounds for the cone"
        LOWER = np.array([18, 44, 101])   # determined experimentally
        UPPER = np.array([31, 232, 255])   # determined experimentally

    class Cube:
        "The lower and upper bounds for the cube"
        LOWER = np.array([118,87,86]) 
        UPPER = np.array([133,255,255])


def process_object(threshold):
    """
    Gets the center of the object from the camera feed.
    """
    for frame in infinite_frame_stream():
        noise_reduction = reduce_noise(frame, threshold.LOWER, threshold.UPPER)
        
        # calculate x,y coordinate of center
        contours,_ = cv.findContours(noise_reduction, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contours = get_contours(noise_reduction)
        if len(contours) == 0:
            print("no contours found")
            continue
        center = maximum_contour_center(contours)
        
        cv.circle(frame, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)

        cv.imshow("frame", frame)
        cv.imshow("noise_reduction", noise_reduction)

def main():
    """
    Processes the object based on the value of CUBE.
    """
    if CUBE:
        process_object(Threshold.Cube)
    else:
        process_object(Threshold.Cone)

if __name__ == "__main__":
    main()
