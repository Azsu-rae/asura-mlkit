
import cvkit.utils as ut
from pathlib import Path
import cv2

imgpath = Path.home() / "data" / "Images"

im1 = cv2.imread("{}/compteur.jpg".format(imgpath), cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No image!")

imc1 = cv2.imread("{}/pepper.bmp".format(imgpath))
if imc1 is None:
    raise ValueError("No image!")

im1_equ = ut.equalization(im1)
im1_norm = ut.normalization(im1)
#im2cv = cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#imc2 = cv2.normalize(imc1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('compteur', im1)
cv2.imshow('normalization', im1_norm)
cv2.imshow('equalization', im1_equ)

# cv2.imshow('pepper', imc1)
# cv2.imshow('imc2', imc2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cvkit.utils.display_gray(cvkit.utils.histo_gray(imc1))

# norm_img = cv2.normalize(im1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# cnorm_img = cvkit.filter.min_max_scalling(im1)
# cv2.imshow('normalized', norm_img)
# cv2.imshow('custom normalized', cnorm_img)
# cv2.imshow('original', im1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
