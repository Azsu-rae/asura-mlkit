
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im2 = cv.imread('Images/pepper.bmp') 
h, w = im2.shape[:2]

blue = np.zeros(256)
green = np.zeros(256)
red = np.zeros(256)
for i in range(h):
    for j in range(w):
        (b, g, r) = im2[i][j]
        blue[b] += 1
        blue[g] += 1
        blue[r] += 1

cv.boundingRect

plt.hist(blue, bins=30, color='b', alpha=0.7)
plt.hist(green, bins=30, color='g', alpha=0.7)
plt.hist(red, bins=30, color='r', alpha=0.7)
plt.show()
