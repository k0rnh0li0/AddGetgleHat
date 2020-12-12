#!/usr/bin/env python3

import sys
import math

import cv2
import numpy as np

# load classifiers
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cascade_eye = cv2.CascadeClassifier("haarcascade_eye.xml")

def display_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_hat(img, hat):
    # image dimensions
    img_h, img_w, img_channels = img.shape
    hat_h, hat_w, hat_channels = hat.shape

    # convert images to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hat_gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)

    # create mask and inverse of hat
    ret, original_mask = cv2.threshold(hat_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)

    # find faces
    faces = cascade_face.detectMultiScale(img_gray, 1.3, 5)

    # add getgle hat to each face
    for (x,y,w,h) in faces:
        face_x1 = x
        face_x2 = x + w
        face_y1 = y
        face_y2 = y + h

        hat_width = int(w*1.2)
        hat_height = int(hat_width * hat_h / hat_w)

        # find slope between eyes to try to correct for head tilt
        eye_tilt = 0
        eye_roi = img[y:y+h,x:x+w]
        eye_roi_gray = img_gray[y:y+h,x:x+w]
        eyes = cascade_eye.detectMultiScale(eye_roi_gray)

        EYE_ADJUST = 0

        if len(eyes) == 2:
            dx = eyes[1][0] - eyes[0][0]
            dy = eyes[1][1] - eyes[0][1]
            EYE_ADJUST = dx

            if dx == 0:
                eye_tilt = math.pi / 2
            else:
                eye_tilt = math.atan(dy / dx)

        # find bounding rect for hat, adjusting for eye tilt
        tilt_dx = int(math.cos(eye_tilt) * EYE_ADJUST)
        tilt_dy = int(math.sin(eye_tilt) * EYE_ADJUST)

        hat_x1 = face_x2 - int(w/2) - int(hat_width/2.25) + tilt_dx
        hat_y1 = face_y1 - int(h*0.4) + tilt_dy

        hat_x2 = hat_x1 + hat_width
        hat_y2 = hat_y1 + hat_height

        # resize hat to head
        hat_res = cv2.resize(hat, (hat_width,hat_height), interpolation = cv2.INTER_AREA)
        mask_res = cv2.resize(original_mask, (hat_width,hat_height), interpolation = cv2.INTER_AREA)
        mask_inv_res = cv2.resize(original_mask_inv, (hat_width,hat_height), interpolation = cv2.INTER_AREA)

        # clip hat img/masks to image edges
        ldx = 0 if hat_x1 >= 0 else 0 - hat_x1
        rdx = 0 if hat_x2 < img_w else hat_x2 - img_w
        tdy = 0 if hat_y1 >= 0 else 0 - hat_y1
        bdy = 0 if hat_y2 < img_h else hat_y2 - img_h

        hat_res = hat_res[tdy:hat_height-bdy,ldx:hat_width-rdx]
        mask_res = mask_res[tdy:hat_height-bdy,ldx:hat_width-rdx]
        mask_inv_res = mask_inv_res[tdy:hat_height-bdy,ldx:hat_width-rdx]

        # apply masks to image
        roi = img[hat_y1+tdy:hat_y2-bdy,hat_x1+ldx:hat_x2-rdx]
        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_res)
        roi_fg = cv2.bitwise_and(hat_res,hat_res,mask=mask_inv_res)
        dst = cv2.add(roi_bg, roi_fg)

        img[hat_y1+tdy:hat_y2-bdy,hat_x1+ldx:hat_x2-rdx] = dst

    return img

if __name__ == "__main__":
    print("I AM K0RNH0LI0!")
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: ./getglehat.py <input img> [output img]")
        exit()

    input_path = sys.argv[1]
    output_path = None

    if len(sys.argv) == 3:
        output_path = sys.argv[2]

    # load images
    img = cv2.imread(input_path)
    hat = cv2.imread("hat.png")

    img = add_hat(img, hat)

    if output_path == None:
        display_img(img)
    else:
        cv2.imwrite(output_path, img)
