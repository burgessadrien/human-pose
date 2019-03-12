import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

CANNY_THRESH_1 =10
CANNY_THRESH_2 = 200

upper_body_cascade = cv2.CascadeClassifier('venv/lib/python3.7/site-packages/cv2/data/haarcascade_upperbody.xml')
lower_body_cascade = cv2.CascadeClassifier('venv/lib/python3.7/site-packages/cv2/data/haarcascade_lowerbody.xml')

def find_body_part(mask, ratioY, ratioX):
    bodyPartAverage = 0
    bodyPart = np.zeros((int(mask.shape[0] / ratioY), (int(mask.shape[1] / ratioX)), 3), np.uint8)
    width = bodyPart.shape[1]
    height = bodyPart.shape[0]
    y, x = 0,0
    for i in range(bodyPart.shape[0], mask.shape[0],1):
        for j in range(bodyPart.shape[1], mask.shape[1],1):
            tmp = mask[(i - height):i, (j - width):j]
            average = np.average(tmp[0:]) # tmp[x] = [255,255,255] when white
            if average > bodyPartAverage:
                bodyPartAverage = average
                y, x = i, j

    bodyPart = mask[(y - height):y,(x - width):x]
    return [bodyPart,[y,x]]


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
        frame = cv2.imread("images/man.jpg")
        #w = cam.get(3)
        #h = cam.get(4)
        #print(w,h)1
        height = frame.shape[0]
        width = frame.shape[1]
        canvas = np.zeros((height, width, 3), np.uint8)
        threshHuman = np.zeros((height, width, 3), np.uint8)
        human = np.zeros((height, width, 3), np.uint8)
        torso = np.zeros((height, width, 3), np.uint8)
        (rects, weights) = hog.detectMultiScale(frame, winStride=(5, 5),padding=(30, 30), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        bodypartMask = frame
        if len(pick) > 0:
            print("doing stuff")
            (xA, yA, xB, yB) = pick[0]
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            human = frame[yA:yB, xA:xB]
            #human = cv2.bilateralFilter(human, 26, 46, 13)
            grayHuman = cv2.cvtColor(human, cv2.COLOR_BGR2GRAY)
            #[a,b,c,d] = upper_body_cascade.detectMultiScale(grayHuman, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE)[0]
            #cv2.rectangle(human,(a,b),(a,b),(255,0,0),2)
            #[f,g,h,i] = lower_body_cascade.detectMultiScale(grayHuman, 1.1 , 3)
            #cv2.rectangle(human,(f,g),(f+h,g+i),(255,0,0),2)
            
            threshHuman = cv2.Canny(human,CANNY_THRESH_1,CANNY_THRESH_2)
            contours,high = cv2.findContours(threshHuman, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c  = sorted(contours, key=cv2.contourArea, reverse=True)
            canvas = np.ones((threshHuman.shape[0], threshHuman.shape[1], 3), np.uint8)
            cv2.drawContours(canvas, c, 1, (255,255,255), -1)

            torso, coor = find_body_part(canvas,2, 2)
            cv2.rectangle(frame, (xA + coor[1] - torso.shape[1], yA + coor[0] - torso.shape[0]),(xA + coor[1],yA + coor[0]), (0, 255, 0), 2)

            bodypartMask = canvas[0:(coor[0] - torso.shape[0]),(coor[1] - torso.shape[1]):coor[1]]
            head,headCoor = find_body_part(bodypartMask, 3, 3)
            cv2.rectangle(frame, ((xA  + coor[1] - torso.shape[1]+ headCoor[1] - head.shape[1]), yA + headCoor[0] - head.shape[0]),((xA  + coor[1] - torso.shape[1]+ headCoor[1]),yA + headCoor[0]), (0, 255, 0), 2)
            
        cv2.imshow('body part', np.array(bodypartMask, np.uint8))
        cv2.imshow('human image', np.array(human, dtype=np.uint8))
        cv2.imshow('human' , np.array(threshHuman, dtype = np.uint8 ))
        cv2.imshow('image', np.array(frame, dtype = np.uint8))
        cv2.imshow('test' , np.array(canvas, dtype = np.uint8 ))

        k = cv2.waitKey(1)

if __name__ == "__main__":
    readVideo()
