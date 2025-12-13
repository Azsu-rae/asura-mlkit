
from pathlib import Path
import cv as cvkit
import cv2

# Q1

imgdirpath = Path.home() / "data" / "Images"

imgpath = "{}/cameraman.bmp".format(imgdirpath)
im1 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No image!")

imgpath = "{}/cameraman_bruit_gauss_sig0_001.bmp".format(imgdirpath)
im2 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
if im2 is None:
    raise ValueError("No image!")

imgpath = "{}/cameraman_bruit_sel_poivre_p_10.bmp".format(imgdirpath)
im3 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
if im3 is None:
    raise ValueError("No image!")

cv2.imshow('original', im1)
cv2.imshow('Gaussian noise', im2)
cv2.imshow('Salt & pepper noise', im3)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Q2

ker = 3
im2f = cv2.blur(im2, (ker, ker))
im2cf = cvkit.filter.mean(im2, ker)

cv2.imshow('gaussian noise (distributed)', im2)
cv2.imshow('filtered gaussian noise', im2f)
cv2.imshow('custorm filtered gaussian noise', im2cf)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Q3

ker = 3
im3fc = cvkit.filter.median(im3, ker)
im3f = cv2.medianBlur(im3, ker)
im2fm = cv2.medianBlur(im2, ker)

cv2.imshow('G noise', im2)
cv2.imshow('SP noise', im3)

cv2.imshow('MedianF G', im2fm)
cv2.imshow('MedianF SP', im3f)
cv2.imshow('CMedianF SP', im3fc)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Q4

psnr_opencv = {}

psnr_opencv['im1_im2'] = cv2.PSNR(im1, im2)
psnr_opencv['im1_im2f'] = cv2.PSNR(im1, im2f)
psnr_opencv['im1_im2fm'] = cv2.PSNR(im1, im2fm)

psnr_opencv['im1_im3'] = cv2.PSNR(im1, im3)
psnr_opencv['im1_im3f'] = cv2.PSNR(im1, im3f)

psnr_custom = {}

psnr_custom['im1_im2'] = cvkit.utils.psnr(im1, im2)
psnr_custom['im1_im2f'] = cvkit.utils.psnr(im1, im2f)
psnr_custom['im1_im2fm'] = cvkit.utils.psnr(im1, im2fm)

psnr_custom['im1_im3'] = cvkit.utils.psnr(im1, im3)
psnr_custom['im1_im3f'] = cvkit.utils.psnr(im1, im3f)

# print(f"\n1. PSNR(im1, im2)    = {psnr_opencv['im1_im2']:.4f} dB")
# print(f"\n2. PSNR(im1, imf2)   = {psnr_opencv['im1_im2f']:.4f} dB")
# print(f"\n5. PSNR(im1, imfm2)  = {psnr_opencv['im1_im2fm']:.4f} dB")
# print(f"\n3. PSNR(im1, im3)    = {psnr_opencv['im1_im3']:.4f} dB")
# print(f"\n4. PSNR(im1, imf3)   = {psnr_opencv['im1_im3f']:.4f} dB")
#
# print(f"\n1. Custom PSNR(im1, im2)    = {psnr_custom['im1_im2']:.4f} dB")
# print(f"\n2. Custom PSNR(im1, imf2)   = {psnr_custom['im1_im2f']:.4f} dB")
# print(f"\n5. Custom PSNR(im1, imfm2)  = {psnr_custom['im1_im2fm']:.4f} dB")
# print(f"\n3. Custom PSNR(im1, im3)    = {psnr_custom['im1_im3']:.4f} dB")
# print(f"\n4. Custom PSNR(im1, imf3)   = {psnr_custom['im1_im3f']:.4f} dB")

# Q5

ker = 5
sigma = 1.0
im2fg = cv2.GaussianBlur(im2, (ker,ker), sigmaX=sigma)
im2fgc = cvkit.filter.gaussian(im2, 5, 1.0)

print(f"\nPSNR(im2fg, im2fgc)   = {cv2.PSNR(im2fg, im2fgc)} dB")

cv2.imshow('gf', im2fg)
cv2.imshow('cgf', im2fgc)
cv2.waitKey(0)
cv2.destroyAllWindows()

d = 5
sigmaColor = 10.0
sigmaSpace = 10.0
im2fb = cv2.bilateralFilter(im2, d, sigmaColor, sigmaSpace)
im2fbc = cvkit.filter.bilateral(im2, d, sigmaSpace, sigmaColor)

print(f"\nPSNR(im2fb, im2fbc)   = {cv2.PSNR(im2fb, im2fbc)} dB")

cv2.imshow('original', im2)
cv2.imshow('bf', im2fb)
cv2.imshow('cbf', im2fbc)
cv2.waitKey(0)
cv2.destroyAllWindows()

# im2fbc = cvkit.filtre_bilateral(im2, d, sigmaColor, sigmaSpace)
# cv2.imshow("G Bilateral", im2fb)
# cv2.imshow("G BL C", im2fbc)
#
# im2fg = cv2.GaussianBlur(im2, (5,5), sigmaX=2.0)
#
# cv2.imshow("G Blur", im2fg)
# cv2.imshow("G Bilateral", im2fb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
