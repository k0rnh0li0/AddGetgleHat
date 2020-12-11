#!/usr/bin/env python3

import sys

import cv2
import numpy as np

def display_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: ./getglehat.py <input img> [output img]")
        exit()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # load face classifier
    cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # load images
    img = cv2.imread(input_path)
    hat = cv2.imread("hat.png")

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

    for (x,y,w,h) in faces:
        face_x1 = x
        face_x2 = x + w
        face_y1 = y
        face_y2 = y + h

        hat_width = int(w*1.2)
        hat_height = int(hat_width * hat_h / hat_w)

        # find bounding rect for hat
        hat_x1 = face_x2 - int(w/2) - int(hat_width/2.25)
        hat_x2 = hat_x1 + hat_width
        hat_y1 = face_y1 - int(h*0.4)
        hat_y2 = hat_y1 + hat_height

        # resize hat to head
        hat_res = cv2.resize(hat, (hat_width,hat_height), interpolation = cv2.INTER_AREA)
        mask_res = cv2.resize(original_mask, (hat_width,hat_height), interpolation = cv2.INTER_AREA)
        mask_inv_res = cv2.resize(original_mask_inv, (hat_width,hat_height), interpolation = cv2.INTER_AREA)

        roi = img[hat_y1:hat_y2,hat_x1:hat_x2]
        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_res)
        roi_fg = cv2.bitwise_and(hat_res,hat_res,mask=mask_inv_res)
        dst = cv2.add(roi_bg, roi_fg)

        img[hat_y1:hat_y2,hat_x1:hat_x2] = dst

    display_img(img)
