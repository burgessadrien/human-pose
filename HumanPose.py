
import cv2
import numpy as np

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
w = cam.get(3)
h = cam.get(4)
print(w,h)
k = cv2.waitKey(1)
while (k%256 != 27):
    k = cv2.waitKey(1)
    ret, frame = cam.read()
    cv2.imshow('test' , np.array(frame, dtype = np.uint8 ))

    
def histogramOfGradients(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)