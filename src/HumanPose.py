from imutils.object_detection import non_max_suppression
from imutils import paths, resize
import numpy as np
import argparse
import imutils
import cv2
from math import cos, sin, atan
import time


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

# pt2 x > pt1 x
def find_angle(pt1, pt2):
    width = pt2[0] - pt1[0]
    height = pt2[1] - pt1[1]
    return  atan(height / width)

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
    theta = 0
    if len(eyes) >= 2:
        eyes = eyes[0:2]
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]
        p1 = (ex1 + ew1//2, ey1 + eh1//2)
        p2 = (ex2 + ew2//2, ey2 + eh2//2)
        cv2.line(face_image, p1, p2, (0, 0, 255), 2)
        theta = find_angle(p1,p2)

    # cv2.imshow("other image", image)
    return faces[0], theta

def verify_top_left_bottom_right(p1,p2):
    t1 = [p1[0], p1[1]]
    t2 = [p2[0], p2[1]]

    if t2[0] < t1[0]:
        tmp = t2[0]
        t2[0] = t1[0]
        t1[0] = tmp

    if t1[1] > t2[1]:
        tmp = t2[1]
        t2[1] = t1[1]
        t1[1] = tmp

    return (t1[0],t1[1]), (t2[0], t2[1])

def get_rectangle_score(image, p1, p2):
    p1,p2 = verify_top_left_bottom_right(p1,p2)
    rectangle = image[p1[1]:p2[1], p1[0]:p2[0]]
    non_zero_sum = cv2.countNonZero(rectangle)
    area = (p2[0] - p1[0])*(p2[1]-p1[1])
    return (area, non_zero_sum/area)


def fit_torso(thresh_image, torso_orig, torso_width, torso_height):
    scale = 0.25
    score_thresh = 0.93
    max_score = 0
    max_area = 0
    best_t1 = (0, 0)
    best_t2 = (0, 0)
    best_theta = 0
    best_img = thresh_image
    best_rotate_orig = (0,0)
    for scale_hund in range(75, 150, 15):
        for theta in range(-40, 40, 5):
            scale = scale_hund/100
            scaled_torso_height = int(scale * torso_height)
            scaled_torso_width = int(scale * torso_width)

            # t1: top left, t2: bottom right
            t1 = (torso_orig[0] - scaled_torso_width//2, torso_orig[1]-scaled_torso_height//2)
            t2 = (torso_orig[0] + scaled_torso_width//2, torso_orig[1]+scaled_torso_height//2)
            rotate_orig = (t1[0] + (t2[0]-t1[0])//2, t1[1])

            rot_img = thresh_image.copy()
            rot_img = rotate_image(rot_img, rotate_orig, theta)
            (area, score) = get_rectangle_score(rot_img, t1, t2)

            if score > score_thresh and area > max_area:
                best_t1 = t1
                best_t2 = t2
                max_score = score
                max_area = area
                best_theta = theta
                best_rotate_orig = rotate_orig

    best_t = rotate_rectangle(best_t1, best_t2, best_theta, torso_orig)
    return (best_t, best_theta, best_rotate_orig)

def fit_limb(thresh_image, limb_orig, limb_width, limb_height, limb_side, theta_begin, theta_end, theta_iter):
    scale = 0.25
    score_thresh = 0.85
    max_score = 0
    max_area = 0
    best_t1 = (0, 0)
    best_t2 = (0, 0)
    best_theta = 0
    best_img = thresh_image

    for scale_height_hund in range(75, 150, 10):
        for scale_width_hund in range(75,150,10):
            for theta in range(theta_begin, theta_end, theta_iter):
                scale_height = scale_height_hund/100
                scale_width = scale_width_hund/100
                scaled_limb_height = int(scale_height * limb_height)
                scaled_limb_width = int(scale_width * limb_width)

                # t1: top left, t2: bottom right
                t1 = limb_orig
                if limb_side == "left":
                    t2 = (limb_orig[0] + scaled_limb_width, limb_orig[1]+scaled_limb_height)
                elif limb_side == "right":
                    t2 = (limb_orig[0] + scaled_limb_width, limb_orig[1]-scaled_limb_height)
                else:
                    t1 = (limb_orig[0] - scaled_limb_width//2, limb_orig[1])
                    t2 = (limb_orig[0] + scaled_limb_width//2, limb_orig[1]+scaled_limb_height)

                rotate_orig = limb_orig
                rot_img = thresh_image.copy()
                rot_img = rotate_image(rot_img, rotate_orig, theta)
                (area, score) = get_rectangle_score(rot_img, t1, t2)
                #cv2.rectangle(rot_img, t1, t2, 150)
                #cv2.imshow("best right", rot_img)
                #cv2.waitKey()

                if score > score_thresh and area > max_area:
                    best_t1 = t1
                    best_t2 = t2
                    max_score = score
                    max_area = area
                    best_theta = theta

    thresh_color = thresh_image.copy()
    best_t = rotate_rectangle(best_t1, best_t2, best_theta, limb_orig)
    return (best_t, best_theta)


def rotate_image(image, rot_orig, angle):
    rot_img = image.copy()
    rot_mat = cv2.getRotationMatrix2D(rot_orig, angle, 1.0)
    result = cv2.warpAffine( rot_img, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_point(pos, orig, angle):
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


def rotate_rectangle(t1, t2, theta, origin):
    x1 = t1[0]
    y1 = t1[1]
    x2 = t2[0]
    y2 = t2[1]

    # top left
    new_t1 = rotate_point((x1, y1), origin, theta)
    # top right
    new_t2 = rotate_point((x2, y1), origin, theta)
    # bottom right
    new_t3 = rotate_point((x2, y2), origin, theta)
    # bottom left
    new_t4 = rotate_point((x1, y2), origin, theta)

    return (new_t1, new_t2, new_t3, new_t4)


def draw_rect(image, points):
    (p1, p2, p3, p4) = points
    cv2.line(image, p1, p2, (0, 0, 255), thickness=2)
    cv2.line(image, p2, p3, (0, 0, 255), thickness=2)
    cv2.line(image, p3, p4, (0, 0, 255), thickness=2)
    cv2.line(image, p4, p1, (0, 0, 255), thickness=2)

# To remove section specified from image
def remove_section(rectangle, thresh_image, theta, rotate_orig):
    t1,t2,t3,t4 = rectangle
    np_rectangle = np.array(rectangle)
    image = cv2.fillConvexPoly(thresh_image, np_rectangle, 0)
    return image

def find_midway(p1,p2):
    return (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

def main():

    set_width = 1500
    # both MOG and MOG2 can be used, with different parameter values
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    person_image = "images/just_john/john2.jpg"
    # apply the algorithm for background images using learning rate > 0
    bgImageFile = "images/just_john/bg.jpg"
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
            p2 = (xB, (yA + yB))
            max_area = area

    cv2.rectangle(image, p1, p2, (0, 255, 0), 2)

    # going to threshold using background subtraction
    # image = cv2.imread(person_image)
    fgmask = backgroundSubtractor.apply(image, learningRate=0)
    ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((17, 17), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel_close = np.ones((35,35), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("masked image", fgmask)

    # finding the face
    # this should probably be done before the torso, than have the torso position linked to
    # the position of the face
    (x, y, w, h), face_theta = find_face(image)
    face_points = ((x, y), (x+w, y+h))
    face_h = h
    face_w = w
    remove_face = (( x, y), ( x+w, y), ( x+w,y+h ), ( x,y+h ))
    face_orig = (x + w//2, y + h//2)
    fgmask = remove_section(remove_face, fgmask, face_theta, face_orig)

    # # now we will find the torso
    # torso, coor = find_body_part(upper_half, 2, 2)
    (xA, yA) = p1
    (xB, yB) = p2

    # remove face from fgmask
    #sfgmask = remove_section((x, y, w, h), fgmask)

    # take a portion of the thresholded image
    # based on the bounding box that we got from the HOOG and top of face
    human = fgmask[p1[1]:p2[1], p1[0]:p2[0]]

    # cv2.rectangle(image, (xA + coor[1] - torso.shape[1], yA + coor[0] -
    #                       torso.shape[0]), (xA + coor[1], yA + coor[0]), (0, 255, 0), 2)

    # draw the face rectangle
    cv2.rectangle(image, face_points[0], face_points[1], (0, 255, 0), 2)

    # finding the torso from the face
    torso_width = int(1.8*face_h)
    torso_height = int(2.5*face_h)

    torso_orig = (x+w//2, y+h + torso_height//2 + 50)
    (tp1, tp2, tp3, tp4), torso_theta, torso_orig = fit_torso(human, (torso_orig[0]-xA, torso_orig[1]-yA), torso_width, torso_height)

    draw_rect(image, ((tp1[0]+xA, tp1[1]+yA), (tp2[0]+xA, tp2[1]+yA), (tp3[0]+xA, tp3[1]+yA), (tp4[0]+xA, tp4[1]+yA)))

    # define shoulder and leg points
    shoulder_right_pt = tp1
    shoulder_left_pt = tp2
    thigh_left_pt = tp4
    thigh_right_pt = tp3

    # Identify right arm of person
    (uar1,uar2,uar3,uar4), uar_theta = fit_limb(human, shoulder_right_pt, int(face_h*1.1), face_h//2, "right", -90, -270, -5)
    draw_rect(image, ( (uar1[0]+xA, uar1[1]+yA), (uar2[0]+xA, uar2[1]+yA), (uar3[0]+xA, uar3[1]+yA), (uar4[0]+xA, uar4[1]+yA) ) )

    # identify left arm of person
    (ual1,ual2,ual3,ual4), ual_theta = fit_limb(human, shoulder_left_pt, int(face_h*1.1), face_h//2, "left", -90, 90, 5)
    draw_rect(image, ( (ual1[0]+xA, ual1[1]+yA), (ual2[0]+xA, ual2[1]+yA), (ual3[0]+xA, ual3[1]+yA), (ual4[0]+xA, ual4[1]+yA) ) )


    # identify right leg of person
    (ulr1,ulr2,ulr3,ulr4), ulr_theta = fit_limb(human, thigh_right_pt, int(face_h*1.5), int(face_h//2), "left", 0, 110, 5)
    draw_rect(image, ( (ulr1[0]+xA, ulr1[1]+yA), (ulr2[0]+xA, ulr2[1]+yA), (ulr3[0]+xA, ulr3[1]+yA), (ulr4[0]+xA, ulr4[1]+yA) ) )

    # identify left leg of person
    (ull1,ull2,ull3,ull4), ull_theta = fit_limb(human, thigh_left_pt, int(face_h*1.5), int(face_h//2), "right", -180, -290, -5)
    draw_rect(image, ( (ull1[0]+xA, ull1[1]+yA), (ull2[0]+xA, ull2[1]+yA), (ull3[0]+xA, ull3[1]+yA), (ull4[0]+xA, ull4[1]+yA) ) )
    
    no_torso = remove_section( (tp1, tp2, tp3, tp4) , human, torso_theta, torso_orig )

    upper_arm_right = remove_section( (uar1,uar2,uar3,uar4) , no_torso, uar_theta, shoulder_right_pt )
    upper_arm_left = remove_section( (ual1,ual2,ual3,ual4) , no_torso, ual_theta, shoulder_left_pt )
    upper_thigh_right = remove_section( (ulr1,ulr2,ulr3,ulr4) , no_torso, ulr_theta, thigh_right_pt )
    upper_thigh_left = remove_section( (ull1,ull2,ull3,ull4) , no_torso, ull_theta, thigh_left_pt )


    cv2.imshow("arm right", upper_arm_right)
    #cv2.imshow("arm left", upper_arm_left)
    #cv2.imshow("leg left", upper_thigh_left)
    #cv2.imshow("leg right", upper_thigh_right)

    # identifying right lower arm
    forearm_right_pt = find_midway(uar3,uar2)
    (lar1,lar2,lar3,lar4), uar_theta = fit_limb(upper_arm_right, forearm_right_pt, int(face_h//2), face_h, "center", 180, -180, -5)
    draw_rect(image, ( (lar1[0]+xA, lar1[1]+yA), (lar2[0]+xA, lar2[1]+yA), (lar3[0]+xA, lar3[1]+yA), (lar4[0]+xA, lar4[1]+yA) ) )

    # identifying left lower arm
    forearm_left_pt = find_midway(ual3,ual2)
    (lal1,lal2,lal3,lal4), ual_theta = fit_limb(upper_arm_left, forearm_left_pt, int(face_h//2), face_h, "center", -180, 180, 5)
    draw_rect(image, ( (lal1[0]+xA, lal1[1]+yA), (lal2[0]+xA, lal2[1]+yA), (lal3[0]+xA, lal3[1]+yA), (lal4[0]+xA, lal4[1]+yA) ) )

    # identifying right lower leg
    shin_right_pt = find_midway(ulr3,ulr2)
    (llr1,llr2,llr3,llr4), ulr_theta = fit_limb(upper_thigh_right, shin_right_pt, int(face_h//2), face_h, "center", 180, -180, -5)
    draw_rect(image, ( (llr1[0]+xA, llr1[1]+yA), (llr2[0]+xA, llr2[1]+yA), (llr3[0]+xA, llr3[1]+yA), (llr4[0]+xA, llr4[1]+yA) ) )

    # identifying left lower leg
    shin_left_pt = find_midway(ull3,ull2)
    (lll1,lll2,lll3,lll4), ull_theta = fit_limb(upper_thigh_left, shin_left_pt, int(face_h//2), face_h, "center", -180, 180, 5)
    draw_rect(image, ( (lll1[0]+xA, lll1[1]+yA), (lll2[0]+xA, lll2[1]+yA), (lll3[0]+xA, lll3[1]+yA), (lll4[0]+xA, lll4[1]+yA) ) )


    # show the original images with rectangles and the thresholded image
    # cv2.imshow("mask", fgmask)
    # cv2.imshow("upper half", human)
    image = resize(image, height=720)
    cv2.imshow("Boxes", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Body Part Approximations:

# head:         width:          height: 1 head
# torso:        width:          height: 3 heads
# upper arm:    width:          height: 1 heads
# lower arms:   width:          height: 1.5 heads
# upper leg:    width:          height: 2 heads
# lower leg:    width:          height: 2 heads
