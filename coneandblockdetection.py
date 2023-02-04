import cv2
import numpy as np
import time

cube = False 
cap = cv2.VideoCapture(1+cv2.CAP_DSHOW)

x_res = 720 #determine later
y_res = 480 #determine later

center_coord = np.array([x_res/2, y_res/2])


"""
cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    #print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.39.52.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

#waits for Network Tables
with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()
#get the table
vision_nt = NetworkTables.getTable('Vision')
"""

if cube:
    lower_threshold_cube = np.array([118,87,86])
    upper_threshold_cube = np.array([133,255,255])
    while True:
        startTime = time.time()
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, lower_threshold_cube, upper_threshold_cube)
        noise_reduction = cv2.blur(thresh,(20,20))
        noise_reduction = cv2.inRange(noise_reduction, 1, 75)
        noise_reduction = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
        M = cv2.moments(noise_reduction)

        contours, hierarchy = cv2.findContours(noise_reduction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_index = 0
        max_area = 0
        coord = np.zeros(2)

        for i,c in enumerate(contours):
            # calculate moments for each contour
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # calculate area of contour
                if(M['m00'] > max_area):
                    max_area = M['m00']
                    coord[0] = cX
                    coord[1] = cY
                    max_index = i
        
        cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (255, 255, 255), -1)
        """
        vision_nt.putNumber('xError', coord[0] - center_coord[0])
        vision_nt.putNumber('yError', center_coord[1] - coord[1])
        vision_nt.putNumber('area', M['m00'])
        """

        # cv2.putText(frame, str(1/(time.time() - startTime)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('frame', frame)
        cv2.imshow('frame', frame)
        cv2.imshow('thresh', thresh)
        cv2.imshow('noise_reduction', noise_reduction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    lower_threshold = np.array([18, 44, 101])   # determined experimentally
    upper_threshold = np.array([31, 232, 255])   # determined experimentally
    
    while True:
        start_time = time.time()

        # getting video frame
        ret, frame = cap.read()

        #convert image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # convert the hsv image to binary image + noise reduction
        thresh = cv2.inRange(hsv, lower_threshold, upper_threshold)
        noise_reduction = cv2.blur(thresh,(20,20))
        noise_reduction = cv2.inRange(noise_reduction, 1, 75)
        noise_reduction = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)

        # calculate moments of binary image
        M = cv2.moments(thresh)
        
        # calculate x,y coordinate of center
        contours, hierarchy = cv2.findContours(noise_reduction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_index = 0
        max_area = 0
        coord = np.zeros(2)

        for i,c in enumerate(contours):
            # calculate moments for each contour
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # calculate area of contour
                if(M['m00'] > max_area):
                    max_area = M['m00']
                    coord[0] = cX
                    coord[1] = cY
                    max_index = i
        
        cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (255, 255, 255), -1)
        """
        vision_nt.putNumber('xError', coord[0] - center_coord[0])
        vision_nt.putNumber('yError', center_coord[1] - coord[1])
        vision_nt.putNumber('area', M['m00'])
        """
        cv2.imshow("result",noise_reduction)
        cv2.imshow("normal",frame)
        #cv2.imshow("normal2",noise_reduction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps = round(1.0 / (time.time() - start_time))
        # print("FPS: ", fps)
