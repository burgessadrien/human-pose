import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def readVideo():
    #cam = cv2.VideoCapture(0)
    #ret, frame = cam.read()
    frame = cv2.imread("..\images\man.jpg")
    #w = cam.get(3)
    #h = cam.get(4)
    #print(w,h)
    k = cv2.waitKey(1)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
       
    while (k%256 != 27):
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        for (xA, yA, xB, yB) in pick:
            #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            human = frame[yA:yB, xA:xB]
            threshHuman = cv2.Canny(human,100,200)
            contours,high = cv2.findContours(threshHuman, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c  = sorted(contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(frame, c, 1, (0,255,0), 3)
        k = cv2.waitKey(1)
        cv2.imshow('test' , np.array(frame, dtype = np.uint8 ))
        #ret, frame = cam.read()

def histogramOfGradients(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = numpy.histogram(angle, bins=9, range=None, normed=None, weights=mag, density=1)

if __name__ == "__main__":
    readVideo()