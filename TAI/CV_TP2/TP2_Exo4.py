
import numpy as np
import cv2

# Q1

im = cv2.imread("Images/Globules_rouges.jpg")
if im is None:
    raise ValueError("No image!")

h, w, channels = im.shape
histo = np.zeros((channels, 256), dtype=np.uint32)

for i in range(h):
    for j in range(w):
        for c in range(channels):
            histo[c, im[i, j]] += 1

lum_histo = 0.299*histo[2] + 0.587*histo[1] + 0.114*histo[0]
lum = 0.299*im[:, :, 2] + 0.587*im[:, :, 1] + 0.114*im[:, :, 0]

max_var = 0
opt_t = int(0)
imsize = h*w
for t in range(256):

    c1 = np.sum(lum_histo[:t])
    c2 = np.sum(lum_histo[t:])

    if c1 == 0 or c2 == 0:
        continue

    p_c1 = c1/imsize
    p_c2 = c2/imsize

    mean_c1 = np.sum(range(t)*lum_histo[:t]) / c1
    mean_c2 = np.sum(range(t, 256)*lum_histo[t:]) / c2

    inter_var = p_c1*p_c2*(mean_c1 - mean_c2)**2
    if max_var < inter_var:
        opt_t = t
        max_var = inter_var


lum[lum < opt_t] = 0
lum[lum >= opt_t] = 255

cv2.imshow("Original", im)
cv2.imshow("Threshold", lum)
cv2.waitKey(0)
