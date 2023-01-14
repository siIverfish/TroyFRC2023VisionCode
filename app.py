import cv2
import numpy as np
import time

cube = False
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

if cube:
    lower_threshold_cube = np.array([118,87,86])
    upper_threshold_cube = np.array([133,255,255])
    while True:
        startTime = time.time()
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, lower_threshold_cube, upper_threshold_cube)
        noise_reduction = cv2.blur(thresh,(20,20))
        noise_reduction = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
        M = cv2.moments(noise_reduction)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(frame, str(1/(time.time() - startTime)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('frame', frame)
        cv2.imshow('frame', frame)
        cv2.imshow('thresh', thresh)
        cv2.imshow('noise_reduction', noise_reduction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    coords = []

    lower_threshold = np.array([18, 76, 101])   # determined experimentally
    upper_threshold = np.array([55, 232, 255])   # determined experimentally
    
    while True:
        start_time = time.time()

        # getting video frame
        ret, frame = cap.read()

        #convert image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # convert the grayscale image to binary image
        thresh = cv2.inRange(hsv, lower_threshold, upper_threshold)
        noise_reduction = cv2.blur(thresh,(20,20))
        noise_reduction = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)

        # calculate moments of binary image
        M = cv2.moments(thresh)
        
        # calculate x,y coordinate of center
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
                # calculate area of contour
                area = M['m00']

                # add coordinates + area to array
                np.append(coords, np.array([cX, cY, area]))

            
        # noise reduction code
        # noise_reduction = cv2.blur(mask,(10,10))
        # noise_reduction = cv2.inRange(noise_reduction,1,75)
        # noise_reduction = cv2.blur(noise_reduction,(15,15))

        max_index = 0
        max_area = 0
        for i in coords:
            if coords[i][2] > max_area:
                max_area = coords[i][2]
                max_index = i


        cv2.imshow("result",noise_reduction)
        cv2.imshow("normal",frame)
        #cv2.imshow("normal2",noise_reduction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps = round(1.0 / (time.time() - start_time))
        # print("FPS: ", fps)
        if max_index < len(coords):
            print("Coords: ", coords[max_index])


        #cv2.imshow("result",noise_reduction)
        #cv2.imshow("normal",frame)
        #cv2.imshow("normal2",noise_reduction)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break