
import cv2
import numpy as np

im1 = cv2.imread('Images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
gaussian_noise = np.zeros(im1.shape, np.float32)
cv2.randn(gaussian_noise, 0, 100)

noisy_img_float = cv2.add(im1.astype(np.float32), gaussian_noise)

im2 = np.clip(noisy_img_float, 0, 255).astype(np.uint8)


cv2.imshow('im1 - Original', im1)
cv2.imshow('im2 - bruit', im2)


cv2.waitKey(0)
cv2.destroyAllWindows()
