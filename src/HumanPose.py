import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

CANNY_THRESH_1 =10
CANNY_THRESH_2 = 200

face_cascade = cv2.CascadeClassifier('venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
upper_body_cascade = cv2.CascadeClassifier('venv/lib/python3.7/site-packages/cv2/data/haarcascade_upperbody.xml')
lower_body_cascade = cv2.CascadeClassifier('venv/lib/python3.7/site-packages/cv2/data/haarcascade_lowerbody.xml')

def readVideo():
    try:
        cam = cv2.VideoCapture(0)
    except:
        print("Unable to open camera feed.")
        exit(1)

    cam.set(3,320)
    cam.set(4,240)
    k = cv2.waitKey(1)
    kernel = np.ones((5,5), np.uint8)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    while (k%256 != 27):
        #ret, frame = cam.read()
        frame = cv2.imread("images/stockperson.jpg")
        #w = cam.get(3)
        #h = cam.get(4)
        #print(w,h)
        width = frame.shape[0]
        height = frame.shape[1]
        canvas = np.zeros((width, height, 3), np.uint8)
        frame = cv2.medianBlur(frame, 5)
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        if len(pick) > 0:
            (xA, yA, xB, yB) = pick[0]
            #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            human = frame[yA:yB, xA:xB]
            grayHuman = cv2.cvtColor(human, cv2.COLOR_BGR2GRAY)
            [x,y,w,h] = face_cascade.detectMultiScale(grayHuman, 1.3, 5)[0]
            cv2.rectangle(human,(x,y),(x+w,y+h),(255,255,0),2)
            [a,b,c,d] = upper_body_cascade.detectMultiScale(grayHuman, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE)[0]
            b = b + y
            d = d - y
            cv2.rectangle(human,(a,b),(a+c,b+d),(255,0,0),2)
            #[f,g,h,i] = lower_body_cascade.detectMultiScale(grayHuman, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE)[0]
            #cv2.rectangle(human,(f,g),(f+h,g+i),(255,0,0),2)

            threshHuman = cv2.Canny(human,CANNY_THRESH_1,CANNY_THRESH_2)
            contours,high = cv2.findContours(threshHuman, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c  = sorted(contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(canvas, c, -1, (255,255,255), -1)
            cv2.imshow('human' , np.array(threshHuman, dtype = np.uint8 ))

        cv2.imshow('image', np.array(frame, dtype = np.uint8))
        cv2.imshow('test' , np.array(canvas, dtype = np.uint8 ))
            #k = cv2.waitKey(1)
        k = cv2.waitKey(1)

if __name__ == "__main__":
    readVideo()
