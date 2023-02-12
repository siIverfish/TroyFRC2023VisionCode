""" Detects the cone / block in the camera stream. """

from cv2 import cv2 as cv
import numpy as np

from common import (
    reduce_noise,
    maximum_contour_center,
    infinite_frame_stream,
)

CUBE = False
PRINTED_CONE = True
CONE = False


class Threshold:
    """
    Holds the lower and upper threshold color values for the objects.
    If the BGR values of a pixel are within these bounds, it will be
    considered part of the object.
    NOTE: Values are in HSV?.
    """

    class Cone:
        "The lower and upper bounds for the cone"
        LOWER = np.array([18, 44, 101])
        UPPER = np.array([31, 232, 255])

    class Cube:
        "The lower and upper bounds for the cube"
        LOWER = np.array([118, 87, 86])
        UPPER = np.array([133, 255, 255])

    class PrintedCone:
        "The lower and upper bounds for a printed image of the cone"
        # middle: [44, 57, 81]
        LOWER = np.array([34, 47, 71])
        UPPER = np.array([54, 67, 91])


def process_object(threshold):
    """
    Given a threshold, this function will process the camera stream
    and detect the object of interest.

    Args:
        threshold (type): An object with LOWER and UPPER attributes, which are RGB values.
    """
    for frame in infinite_frame_stream():
        noise_reduction = reduce_noise(frame, threshold.LOWER, threshold.UPPER)

        # calculate x,y coordinate of center
        contours, _ = cv.findContours(
            noise_reduction,
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE
        )

        center = maximum_contour_center(contours)

        if center is None:
            print("No contours found")
        else:
            cv.circle(frame, (int(center[0]), int(
                center[1])), 5, (255, 255, 255), -1)

        cv.imshow("frame", frame)
        cv.imshow("noise_reduction", noise_reduction)


def main():
    """
    Processes the object based on the value of CUBE.
    """
    if sum([CUBE, CONE, PRINTED_CONE]) != 1:
        print("Please set exactly one of CUBE, CONE, or PRINTED_CONE to True")
        exit(0)

    if CUBE:
        process_object(Threshold.Cube)
    elif PRINTED_CONE:
        process_object(Threshold.PrintedCone)
    elif CONE:
        process_object(Threshold.Cone)


if __name__ == "__main__":
    main()
