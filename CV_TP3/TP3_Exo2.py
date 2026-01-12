
import numpy as np
import cv2

im = cv2.imread("Images/cameraman.bmp", cv2.IMREAD_GRAYSCALE)
if im is None:
    raise ValueError("No image found!")

mask = np.array([-1, 0, 1], dtype=np.float64)
im_f = im.astype(np.float64)

# classical Central Difference Operation
G_x_s = im_f[:, 2:] - im_f[:, :-2]
G_y_s = im_f[2:, :] - im_f[:-2, :]

cv2.imshow("original", im)
cv2.imshow("G_x sliced", G_x_s.astype(np.uint8))
cv2.imshow("G_y sliced", G_y_s.astype(np.uint8))
cv2.waitKey(0)
