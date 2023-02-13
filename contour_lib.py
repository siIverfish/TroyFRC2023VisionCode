""" Useful functions for working with points and contours. """

import numpy as np
import cv2 as cv


def get_maximum_contour(contours):
    """ Returns the contour with the largest area in the list of contours or None if there are no contours. """
    contours = [
        contour for contour in contours \
        if 1_000 < cv.contourArea(contour) < 300_000
    ]
    if len(contours) == 0:
        return None
    return max(contours, key=cv.contourArea)


def get_contour_center(contour):
    """ Returns the center of the contour. """
    moments = cv.moments(contour)
    # No one knows what this does, but it works.
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return np.array([center_x, center_y])


def distance_squared(point1, point2):
    """Returns the distance between two points, but squared for performance because we only need the distance for comparison."""
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def get_angle(point1, point2):
    """Returns the angle from the first point to the second point in degrees."""
    degrees = np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
    while degrees < 0:
        degrees += 360
    # Make the degrees negative to make the angle counterclockwise
    # Resulting in the following:
    #        90
    #    180    0
    #       270
    return -degrees + 360


def get_farthest_point(contour, center=None):
    """Returns the farthest point from the center of the contour."""
    if center is None:
        center = get_contour_center(contour)
    return max(
        (point[0] for point in contour),
        key=lambda point: distance_squared(center, point),
    )


def get_orientation(contour, center=None):
    """Returns the angle of the farthest point from the center of the contour."""
    farthest_point = get_farthest_point(contour, center)
    return angle(center, farthest_point)
