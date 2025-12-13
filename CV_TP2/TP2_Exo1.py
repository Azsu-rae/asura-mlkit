
import cvkit
import cv2

# Q1

from pathlib import Path
imgdir = Path.home() / "data" / "Images"

im1 = cv2.imread("{}/cameraman.bmp".format(imgdir), cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No image!")

# Q2

mean = 0
standard_deviation = 10  # σ (standard deviation) = √100 → σ² (variance) = 100
imgn = cvkit.noise.gaussian(im1, mean, standard_deviation)

cv2.imshow('originale', im1)
cv2.imshow('Gaussian noise', imgn)
cv2.imwrite('{}/cameraman_bruit_gauss_sig0_001.bmp'.format(imgdir), imgn)

# Q3

prob = 0.10
imsp = cvkit.noise.salt_pepper(im1, prob)

cv2.imshow("Salt & Pepper noise", imsp)
cv2.imwrite('{}/cameraman_bruit_sel_poivre_p_10.bmp'.format(imgdir), imsp)

cv2.waitKey(0)
cv2.destroyAllWindows()
