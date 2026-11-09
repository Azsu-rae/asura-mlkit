
import cvkit
import cv2

# Q1

im1 = cv2.imread("Images/cameraman.bmp", cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No image!")

im2 = cv2.imread("Images/cameraman_bruit_gauss_sig0_001.bmp", cv2.IMREAD_GRAYSCALE)
if im2 is None:
    raise ValueError("No image!")

im3 = cv2.imread("Images/cameraman_bruit_sel_poivre_p_10.bmp", cv2.IMREAD_GRAYSCALE)
if im3 is None:
    raise ValueError("No image!")

im2f = cvkit.filter.nagao(im2)
im3f = cvkit.filter.nagao(im3)

cv2.imshow('original', im1)
cv2.imshow('Gaussian noise', im2)
cv2.imshow('Salt & pepper noise', im3)

cv2.imshow('Nagao filtered Gaussian noise', im2f)
cv2.imshow('Nagao filtered Salt & pepper noise', im3f)

cv2.waitKey(0)
cv2.destroyAllWindows()
