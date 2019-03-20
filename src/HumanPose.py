from imutils.object_detection import non_max_suppression
from imutils import paths, resize
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


def find_face(image):
    head_casc = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + head_casc)
    img_cpy = image.copy()

    gray_image = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5
    )

    eye_casc = "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_casc)
    (x, y, w, h) = faces[0]
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes) >= 2:
        print("eyes!")
        eyes = eyes[0:2]
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]
        p1 = (ex1 + ew1//2, ey1 + eh1//2)
        p2 = (ex2 + ew2//2, ey2 + eh2//2)
        cv2.line(roi_color, p1, p2, (0, 0, 255), 2)

    cv2.imshow("other image", image)
    return faces[0]


def get_rectangle_score(thresh_image, p1, p2):
    rectangle = thresh_image[p1[1]:p2[1], p1[0]:p2[0]]
    non_zero_sum = cv2.countNonZero(rectangle)
    area = (p2[0] - p1[0])*(p2[1]-p1[1])
    return (area, non_zero_sum/area)


def fit_torso(thresh_image, image, torso_orig, torso_width, torso_height):
    scale = 0.25
    score_thresh = 0.8
    max_score = 0
    max_area = 0
    best_t1 = (0, 0)
    best_t2 = (0, 0)
    for scale_hund in range(25, 150, 25):
        scale = scale_hund/100
        scaled_torso_height = int(scale * torso_height)
        scaled_torso_width = int(scale * torso_width)
        t1 = (torso_orig[0] - scaled_torso_width//2,
              torso_orig[1]-scaled_torso_height//2)
        t2 = (torso_orig[0] + scaled_torso_width//2,
              torso_orig[1]+scaled_torso_height//2)
        (area, score) = get_rectangle_score(thresh_image, t1, t2)
        if score > score_thresh and area > max_area:
            best_t1 = t1
            best_t2 = t2
            max_score = score
            # for scale in range(0.25, 1.5, 0.25):
    thresh_color = thresh_image.copy()
    thresh_color = cv2.cvtColor(thresh_color, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(thresh_color, best_t1, best_t2, (255, 0, 0), 2)
    cv2.imshow("thresh color", thresh_color)
    return


def main():

    set_width = 1500
    # both MOG and MOG2 can be used, with different parameter values
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True)
    person_image = "images/adrien.jpg"
    # apply the algorithm for background images using learning rate > 0
    bgImageFile = "images/background.jpg"
    bg = cv2.imread(bgImageFile)
    bg = resize(bg, width=set_width)
    for i in range(1, 16):
        backgroundSubtractor.apply(bg, learningRate=0.5)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = cv2.imread(person_image)
    image = resize(image, width=set_width)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # get the largest person detected
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

    # going to threshold using background subtraction
    # image = cv2.imread(person_image)
    fgmask = backgroundSubtractor.apply(image, learningRate=0)
    ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((11, 11), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # take a portion of the thresholded image
    # based on the bounding box that we got from the HOOG
    upper_half = fgmask[p1[1]:p2[1], p1[0]:p2[0]]

    # finding the face
    # this should probably be done before the torso, than have the torso position linked to
    # the position of the face
    (x, y, w, h) = find_face(image)
    face_points = ((x, y), (x+w, y+h))
    face_h = h
    face_w = w

    # # now we will find the torso
    # torso, coor = find_body_part(upper_half, 2, 2)
    (xA, yA) = p1
    (xB, yB) = p2
    # cv2.rectangle(image, (xA + coor[1] - torso.shape[1], yA + coor[0] -
    #                       torso.shape[0]), (xA + coor[1], yA + coor[0]), (0, 255, 0), 2)

    # draw the face rectangle
    cv2.rectangle(image, face_points[0], face_points[1], (0, 255, 0), 2)

    # finding the torso from the face
    torso_width = int(1.8*face_h)
    torso_height = int(2.5*face_h)

    torso_orig = (x+w//2, y+h + torso_height//2 + 50)

    fit_torso(
        upper_half, image, (torso_orig[0]-xA, torso_orig[1]-yA), torso_width, torso_height)

    # show the original images with rectangles and the thresholded image
    cv2.imshow("mask", fgmask)
    cv2.imshow("upper half", upper_half)
    cv2.imshow("Boxes", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
