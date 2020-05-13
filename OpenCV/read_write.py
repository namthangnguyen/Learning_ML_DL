import cv2
import matplotlib.pyplot as plt

# ảnh màu OpenCV tổ chức dữ liệu theo Height, Width, Channel
# Channel được sắp xếp theo thứ tự Blue, Green, Red

# read image
img = cv2.imread('data/jennie.jpg') 

# show image
cv2.imshow('show jennie.jpg with cv2', img)
# plt.imshow(img)
# NOTE: matplotlib bị sai màu do cv2 đọc ảnh màu theo thứ tự BGR
# -> muốn show đúng ta phải chuyển về RGB bằng img = img[:,:,::-1] or
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# image dimension
print('img.shape: ', img.shape)
# read image without red channel
img[:, :, 1] = 0

# write / save modified image
cv2.imwrite('data/pink_jennie.jpg', img)
