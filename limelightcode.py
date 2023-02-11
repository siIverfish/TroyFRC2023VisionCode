import cv2 as cv
import numpy as np

from common import reduce_noise, should_invert, maximum_contour

invert_angle = False
previous_angle = None


def runPipeline(image, llrobot):
    #constants for code
    lower_threshold = np.array([15, 0, 120])    # determined experimentally
    upper_threshold = np.array([31, 255, 255])  # determined experimentally

    #initialize variables in case they return nothing
    llpython = [0,0,0]

    noise_reduction = reduce_noise(image, lower_threshold, upper_threshold)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return np.array([[]]),image,llpython

    max_area_contour = maximum_contour(contours)
    M = cv.moments(max_area_contour)

    # premature return if stuff is wrong
    if M['m00'] == 0 or len(max_area_contour) <= 5:
        return max_area_contour,image,llpython

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    llpython[1] = cX
    llpython[2] = cY

    # draw the contour and center of the shape on the image
    cv.circle(image, (cX, cY), 7, (0, 0, 0), -1)

    leftmost   = tuple(max_area_contour[max_area_contour[:,:,0].argmin()][0])
    rightmost  = tuple(max_area_contour[max_area_contour[:,:,0].argmax()][0])
    topmost    = tuple(max_area_contour[max_area_contour[:,:,1].argmin()][0])
    bottommost = tuple(max_area_contour[max_area_contour[:,:,1].argmax()][0])

    for point in [leftmost, rightmost, topmost, bottommost]:
        cv.circle(image, point, 5, (0, 0, 0), 2)

    _,_,angle = cv.fitEllipse(max_area_contour)

    count = sum(
        1 for point in [leftmost, rightmost, topmost, bottommost] \
        if point[1] > cY
    )

    if should_invert(angle, count):
        angle += 180

    image = cv.putText(image, str(round(angle)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)    
    llpython[0] = angle

    return max_area_contour,image,llpython

