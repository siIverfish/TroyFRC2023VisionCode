import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
 
DELTAVALUE = 20 # variation for how wide the angle "upright" can be, in each direction
def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  #q[0] = p[0] - scale * hypotenuse * cos(angle)
  #q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  max_area_contourr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv.circle(img, max_area_contourr, 3, (255, 0, 255), 2)
  p1 = (max_area_contourr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], max_area_contourr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0]) # ref line
  p2 = (max_area_contourr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], max_area_contourr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0]) # the angle of the thingy
  drawAxis(img, max_area_contourr, p1, (255, 255, 0), 1)
  drawAxis(img, max_area_contourr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
  label = "  Rotation Angle: " + str(int(np.rad2deg(angle))) + " degrees"
  textbox = cv.rectangle(img, (max_area_contourr[0], max_area_contourr[1]-25), (max_area_contourr[0] + 250, max_area_contourr[1] + 10), (255,255,255), -1)
  cv.putText(img, label, (max_area_contourr[0], max_area_contourr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return angle
 
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

    #convert image to HSV
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
        if area < 1000 or 100000 < area:
            continue
        
        #india

        M = cv.moments(c)
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
        ave_y = (leftmost[1] + rightmost[1] + topmost[1] + bottommost[1]) / 4
        if previous_angle is None or abs(angle - previous_angle) > 90:
            if ave_y > cY:
                invert_angle = True
            else:
                invert_angle = False
        
        if invert_angle:
            print (angle + 180)
        else:
            print (angle)

        previous_angle = angle
        
        # Find the orientation of each shape
        #angle = getOrientation(max_area_contour, img)
        #print (returnOrientation(np.rad2deg(angle))) # function to print out if upright or not

    cv.imshow('Output Image', img)
    cv.imshow('Noise Reduction', noise_reduction)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# Save the output image to the current directory
# cv.imwrite("output_img.jpg", img)