import os
import cv2


def convert_to_binary(img_grayscale, thresh=128):
    thresh, binary_img = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
    return binary_img


input_img_path = 'data/jennie.jpg'
# read color imgae with grayscale flag: cv2.IMREAD_GRAYSCALE
img_gray = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

img_binary = convert_to_binary(img_gray, thresh=128)
cv2.imwrite('data/binary_%s' % os.path.basename(input_img_path), img_binary)
