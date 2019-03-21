from imutils.object_detection import non_max_suppression
from imutils import paths, resize
import numpy as np
import argparse
import imutils
import cv2
from math import cos, sin


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
    face_image = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes) >= 2:
        print("eyes!")
        eyes = eyes[0:2]
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]
        p1 = (ex1 + ew1//2, ey1 + eh1//2)
        p2 = (ex2 + ew2//2, ey2 + eh2//2)
        cv2.line(face_image, p1, p2, (0, 0, 255), 2)

    cv2.imshow("other image", image)
    return faces[0]


def get_rectangle_score(thresh_image, p1, p2):
    rectangle = thresh_image[p1[1]:p2[1], p1[0]:p2[0]]
    non_zero_sum = cv2.countNonZero(rectangle)
    area = (p2[0] - p1[0])*(p2[1]-p1[1])
    return (area, non_zero_sum/area)


def fit_torso(thresh_image, image, torso_orig, torso_width, torso_height):
    scale = 0.25
    score_thresh = 0.9
    max_score = 0
    max_area = 0
    best_t1 = (0, 0)
    best_t2 = (0, 0)
    best_theta = 0
    for scale_hund in range(25, 150, 25):
        for theta in range(-40, 40, 5):
            scale = scale_hund/100
            scaled_torso_height = int(scale * torso_height)
            scaled_torso_width = int(scale * torso_width)
            t1 = (torso_orig[0] - scaled_torso_width//2,
                  torso_orig[1]-scaled_torso_height//2)
            t2 = (torso_orig[0] + scaled_torso_width//2,
                  torso_orig[1]+scaled_torso_height//2)
            rot_img = thresh_image.copy()
            rot_img = rotate_image(
                rot_img, (t1[0] + (t2[0]-t1[0])//2, t1[1]), theta)
            (area, score) = get_rectangle_score(rot_img, t1, t2)
            if score > score_thresh and area > max_area:

                best_t1 = t1
                best_t2 = t2
                max_score = score
                max_area = area
                best_theta = theta
                best_img = rot_img.copy()
    print(max_score, max_area)
    thresh_color = thresh_image.copy()
    thresh_color = cv2.cvtColor(thresh_color, cv2.COLOR_GRAY2BGR)
    best_img = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow('best image', best_img)
    return rotate_rectangle(thresh_color, best_t1, best_t2, best_theta)

    # cv2.rectangle(thresh_color, best_t1, best_t2, (255, 0, 0), 2)


def rotate_image(image, rot_orig, angle):
    rot_img = image.copy()
    rot_mat = cv2.getRotationMatrix2D(rot_orig, angle, 1.0)
    result = cv2.warpAffine(
        rot_img, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# def rotate_image(image, angle):
#     if angle == 0:
#         return image
#     height, width = image.shape[:2]
#     rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
#     result = cv2.warpAffine(
#         image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
#     return result


def rotate_point(pos, orig, img, angle):
    if angle == 0:
        return pos
    angle = np.radians(angle)
    x = pos[0]
    y = pos[1]
    orig_x = orig[0]
    orig_y = orig[1]

    newx = cos(angle)*(x-orig_x) - sin(angle)*(y-orig_y)+orig_x
    newy = sin(angle) * (x-orig_x) + cos(angle)*(y-orig_y) + orig_y

    return int(newx), int(newy)


def rotate_rectangle(image, t1, t2, theta):
    x1 = t1[0]
    y1 = t1[1]
    x2 = t2[0]
    y2 = t2[1]

    origin = (x1 + (x2-x1)//2, y1)
    new_t1 = rotate_point((x1, y1), origin, image, theta)
    new_t2 = rotate_point((x2, y1), origin, image, theta)
    new_t3 = rotate_point((x2, y2), origin, image, theta)
    new_t4 = rotate_point((x1, y2), origin, image, theta)

    return (new_t1, new_t2, new_t3, new_t4)


def draw_rect(image, points):
    (p1, p2, p3, p4) = points
    cv2.line(image, p1, p2, (0, 0, 255), thickness=2)
    cv2.line(image, p2, p3, (0, 0, 255), thickness=2)
    cv2.line(image, p3, p4, (0, 0, 255), thickness=2)
    cv2.line(image, p4, p1, (0, 0, 255), thickness=2)

# To remove section specified from image
def removeSection(to_remove, image):
    (x,y,w,h) = to_remove
    mask = np.uint8(np.full(image.shape[:2],255))
    mask[y:y+h,x:x+w] = 0
    return cv2.bitwise_and(image, image, mask = mask)


def main():

    set_width = 1500
    # both MOG and MOG2 can be used, with different parameter values
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    person_image = "images/adrien.jpg"
    # apply the algorithm for background images using learning rate > 0
    bgImageFile = "images/background.jpg"
    bg = cv2.imread(bgImageFile)
    bg = resize(bg, width=set_width)

    # applying background subtraction
    for i in range(1, 16):
        backgroundSubtractor.apply(bg, learningRate=0.5)

    # creating HOG
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = cv2.imread(person_image)
    image = resize(image, width=set_width)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
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

    # remove face from fgmask
    #fgmask = removeSection((x, y, w, h), fgmask)

    # take a portion of the thresholded image
    # based on the bounding box that we got from the HOOG and top of face
    upper_half = fgmask[p1[1]:p2[1], p1[0]:p2[0]]

    # cv2.rectangle(image, (xA + coor[1] - torso.shape[1], yA + coor[0] -
    #                       torso.shape[0]), (xA + coor[1], yA + coor[0]), (0, 255, 0), 2)

    # draw the face rectangle
    cv2.rectangle(image, face_points[0], face_points[1], (0, 255, 0), 2)

    # finding the torso from the face
    torso_width = int(1.8*face_h)
    torso_height = int(2.5*face_h)

    torso_orig = (x+w//2, y+h + torso_height//2 + 50)

    (tp1, tp2, tp3, tp4) = fit_torso(upper_half, image, (torso_orig[0]-xA, torso_orig[1]-yA), torso_width, torso_height)

    # remove torso and face from upper_half 
    upper_half = removeSection( ( tp1[0], 0, tp2[1], tp2[0] + tp1[1] ) , upper_half)

    # define shoulder and leg points
    #shoulder_left_pt = ( tp1[0], tp1[1] )
    #shoulder_right_pt = ( tp1[0] + tp1[1], tp1[1] )
    #image[tp1[1]+yA, tp1[0]+xA] = [255,255,255]
   # image[tp1[1] +yA, tp1[0] + tp1[1]+xA] = [255,255,255]

    draw_rect(image, ((tp1[0]+xA, tp1[1]+yA), (tp2[0]+xA, tp2[1]+yA), (tp3[0]+xA, tp3[1]+yA), (tp4[0]+xA, tp4[1]+yA)))

    # show the original images with rectangles and the thresholded image
    cv2.imshow("mask", fgmask)
    cv2.imshow("upper half", upper_half)
    cv2.imshow("Boxes", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
