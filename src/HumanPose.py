import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
CANNY_THRESH_1 =10
CANNY_THRESH_2 = 200
def readVideo():
    try:
        cam = cv2.VideoCapture(0)
    except:
        print("Unable to open camera feed.")
        exit(1)

    cam.set(3,320)
    cam.set(4,240)
    k = cv2.waitKey(1)
    kernel = np.ones((8,2), np.uint8)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #ret, frame = cam.read()
    frame = cv2.imread("..\images\man.jpg")
    #w = cam.get(3)
    #h = cam.get(4)
    #print(w,h)
    width = frame.shape[0]
    height = frame.shape[1]
    canvas = np.zeros((width, height, 3), np.uint8)
    

    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        human = frame[yA:yB, xA:xB]
        threshHuman = cv2.Canny(human,CANNY_THRESH_1,CANNY_THRESH_2)        
        contours,high = cv2.findContours(threshHuman, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        c  = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(canvas, c, 1, (0,255,0), 1)
    while (k%256 != 27):
        cv2.imshow('image', np.array(frame, dtype = np.uint8))
        cv2.imshow('canvas', np.array(threshHuman, dtype = np.uint8))
        cv2.imshow('human' , np.array(human, dtype = np.uint8 ))
        cv2.imshow('test' , np.array(canvas, dtype = np.uint8 ))
        k = cv2.waitKey(1)


        
        
        #ret, frame = cam.read()


if __name__ == "__main__":
    readVideo()
