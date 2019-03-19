from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def find_body_part(mask, ratio_y, ratio_x):
    bodyPartAverage = 0
    bodyPart = np.zeros(
        (int(mask.shape[0] / ratio_y), (int(mask.shape[1] / ratio_x)), 3), np.uint8)
    width = bodyPart.shape[1]
    height = bodyPart.shape[0]
    y, x = 0, 0
    for i in range(bodyPart.shape[0], mask.shape[0], 1):
        for j in range(bodyPart.shape[1], mask.shape[1], 1):
            tmp = mask[(i - height):i, (j - width):j]
            average = np.average(tmp[0:])  # tmp[x] = [255,255,255] when white
            if average > bodyPartAverage:
                bodyPartAverage = average
                y, x = i, j

    bodyPart = mask[(y - height):y, (x - width):x]
    return [bodyPart, [y, x]]


# both MOG and MOG2 can be used, with different parameter values
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# apply the algorithm for background images using learning rate > 0
for i in range(1, 16):
    bgImageFile = "images/background.jpg"
    bg = cv2.imread(bgImageFile)
    backgroundSubtractor.apply(bg, learningRate=0.5)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
image = cv2.imread('images/adrien.jpg')
orig = image.copy()
# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), scale=1.05)

# draw the original bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
max_area = 0
max_rectangle = (0, 0, 0, 0)
p1 = (0, 0)
p2 = (0, 0)
for (xA, yA, xB, yB) in pick:
    area = (xB - xA) * (yB - yA)
    if area > max_area:
        p1 = (xA, yA)
        p2 = (xB, (yA + 2*yB)//3)
        max_area = area
cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
# apply the algorithm for detection image using learning rate 0
stillFrame = cv2.imread("images/adrien.jpg")
fgmask = backgroundSubtractor.apply(stillFrame, learningRate=0)
ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
upper_half = fgmask[p1[1]:p2[1], p1[0]:p2[0]]

torso, coor = find_body_part(upper_half, 2, 2)
(xA, yA) = p1
(xB, yB) = p2
cv2.rectangle(stillFrame, (xA + coor[1] - torso.shape[1], yA + coor[0] -
                           torso.shape[0]), (xA + coor[1], yA + coor[0]), (0, 255, 0), 2)

# show both images
cv2.imshow("AfterNMS", image)
cv2.imshow("original", stillFrame)
cv2.imshow("mask", fgmask)
cv2.imshow("Upper half", upper_half)
cv2.waitKey()
cv2.destroyAllWindows()
