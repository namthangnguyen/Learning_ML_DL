import os
import cv2


def convert_to_binary(img_grayscale):
    # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    img_binary = cv2.adaptiveThreshold(img_grayscale,
                                       maxValue=255,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=cv2.THRESH_BINARY,
                                       blockSize=15,
                                       C=8)
    return img_binary


input_img_path = 'data/jennie.jpg'
# read color image with grayscale flag: "cv2.IMREAD_GRAYSCALE"
img_grayscale = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

img_binary = convert_to_binary(img_grayscale)
cv2.imwrite('data/adaptive_%s' % os.path.basename(input_img_path), img_binary)
