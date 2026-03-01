
import cv2

im1 = cv2.imread("Images/compteur.jpg", cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No image!")

otsu_threshold, im2 = cv2.threshold(
    im1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('original', im1)
cv2.imshow('binarized', im2)
cv2.waitKey(0)
