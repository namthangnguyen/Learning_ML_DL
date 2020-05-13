import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/jennie.jpg')


''' Box Filter '''
blur_img = cv2.blur(img, ksize=(9, 9)) # or cv2.boxFiler

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
[axi.set_axis_off() for axi in axes.ravel()] # set all axes off
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1].imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
# plt.show()


''' Gaussian Filter''' 
gaussian1 = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=1, sigmaY=1)
gaussian2 = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=4, sigmaY=4)
gaussian3 = cv2.GaussianBlur(img, ksize=(27, 27), sigmaX=1, sigmaY=1)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
[axi.set_axis_off() for axi in axes.ravel()]
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('origin')
axes[0, 1].imshow(cv2.cvtColor(gaussian1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('(9, 9), sigmaX=1, sigmaY=1')
axes[1, 0].imshow(cv2.cvtColor(gaussian2, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('(9, 9), sigmaX=4, sigmaY=4')
axes[1, 1].imshow(cv2.cvtColor(gaussian3, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('(27, 27), sigmaX=1, sigmaY=1')
# plt.show()
# -> Blur Gaussian Filter không quan tâm lắm đến kernel size


'''Median Filter'''
median_img = cv2.medianBlur(img, 9)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
[axi.set_axis_off() for axi in axes.ravel()] # set all axes off
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1].imshow(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))
plt.show()