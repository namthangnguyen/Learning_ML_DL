import cv2
import numpy as np

img = cv2.imread('data/jennie.jpg')
# shape (768, 768, 3)

# resize
# theo kich thuoc co dinh
width = 800
height = 400
img_resized = cv2.resize(src=img, dsize=(width, height))
# rút gọn: cv2.resize(img, (width, height))
img_name = 'jennie_%dx%d.jpg' % (width, height)
cv2.imwrite('data/' + img_name, img_resized)

# theo ti le
fx = 0.5
fy = 1.0
img_resized = cv2.resize(src=img, dsize=None, fx=fx, fy=fy)
# muốn dùng theo tỉ lệ phải chỉ rõ: dsize=None và phải có tên tham số fx, fy
img_name = 'jennie_fx=%.1f_fy=%.1f.jpg' % (fx, fy)
cv2.imwrite('data/' + img_name, img_resized)

# crop image
img_crop = img[50:500, 100:500, :]
crop_name = 'jennie_crop.jpg'
cv2.imwrite('data/' + crop_name, img_crop)

# padding 
img_pad = np.zeros([1068, 1068, 3])
img_pad += 100 # set background is grey
img_pad[150: 918, 150: 918] = img

cv2.imwrite('data/jennie_padding.jpg', img_pad)