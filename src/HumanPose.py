import cv2
import numpy as np

def readVideo():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    w = cam.get(3)
    h = cam.get(4)
    print(w,h)
    k = cv2.waitKey(1)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
       
    while (k%256 != 27):
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        k = cv2.waitKey(1)
        cv2.imshow('test' , np.array(frame, dtype = np.uint8 ))
        ret, frame = cam.read()

def histogramOfGradients(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = numpy.histogram(angle, bins=9, range=None, normed=None, weights=mag, density=1)

if __name__ == "__main__":
    readVideo()