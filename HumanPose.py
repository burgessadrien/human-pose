
import cv2
import numpy as np

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
k = cv2.waitKey(1)
while (k%256 != 27):
    k = cv2.waitKey(1)
    ret, frame = cam.read()
    cv2.imshow('test' , np.array(frame, dtype = np.uint8 ))
