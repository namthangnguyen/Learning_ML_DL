import os
import cv2


def apply_roi(img, roi):
    # resize ROI to match the original image size
    roi = cv2.resize(src=roi, dsize=(img.shape[1], img.shape[0])) # dsize=(width, height)
    
    # scale ROI to [0, 1] => binary mask
    thresh, roi = cv2.threshold(roi, thresh=128, maxval=1, type=cv2.THRESH_BINARY)
    
    # apply ROI on the original image
    new_img = img * roi
    return new_img


input_img_path = 'data/jennie.jpg'
roi_img_path = 'data/ROI.png'

img = cv2.imread(input_img_path)
roi = cv2.imread(roi_img_path)

new_img = apply_roi(img, roi)

cv2.imwrite('data/roi_%s' % os.path.basename(input_img_path), new_img)