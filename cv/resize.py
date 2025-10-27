
import cv2
import numpy as np

im1 = cv2.imread('Images/pepper.bmp', cv2.IMREAD_GRAYSCALE)
h, w = im1.shape[:2]
print(f"Original size: width = {w}, height = {h}")

result = np.zeros((h//2, w//2), dtype=np.uint8)
test = np.zeros((h//2, w//2), dtype=np.uint8)

for i, ii in zip(range(0, h, 2), range(h//2)):
    for j, jj in zip(range(0, w, 2), range(w//2)):
        result[ii][jj] += (int(im1[i][j]) + int(im1[i+1][j]) + int(im1[i][j+1]) + int(im1[i+1][j+1]))//4
        test[ii][jj] += ((im1[i][j]) + (im1[i+1][j]) + (im1[i][j+1]) + (im1[i+1][j+1]))//4

cv2.imshow('im1 - Original', im1)
cv2.imshow('im2 - Resized', result)
cv2.imshow('testing', test)
cv2.waitKey(0)
cv2.destroyAllWindows()
