import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/phongcanh.jpg', 0)
img_equal_hist = cv2.equalizeHist(img)

f, axes = plt.subplots(2,2, figsize=(20, 20))
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('origin')
axes[0, 1].imshow(cv2.cvtColor(img_equal_hist, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('hist equal')
axes[1, 0].hist(img.flatten(), 256, [0, 256])
axes[1, 1].hist(img_equal_hist.flatten(), 256, [0, 256])
plt.show()
plt.close()