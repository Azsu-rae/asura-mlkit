
import numpy as np
import cvkit
import cv2

im = cv2.imread("Images/cameraman.bmp", cv2.IMREAD_GRAYSCALE)
if im is None:
    raise ValueError("No image found!")

"""
1) Edge detection
"""


def operate_mask(mask):
    G_x, G_y = cvkit.segment.gradient(im, mask)

    orientation = cvkit.util.orientation(G_x, G_y)
    norme = cvkit.util.norme(G_x, G_y)

    thin = cvkit.segment.non_maximum_suppression(norme, orientation)
    hysteresis = cvkit.segment.hysteresis(thin)

    return hysteresis.astype(np.uint8)


# cv2.imshow("Original", cvkit.util.normalize(im, 255).astype(np.uint8))
# cv2.imshow("Robert", operate_mask(cvkit.util.robert_mask))
# cv2.imshow("Prewitt", operate_mask(cvkit.util.prewitt_mask))
# cv2.imshow("Sobel", operate_mask(cvkit.util.sobel_mask))
# cv2.waitKey(0)

"""
Operator, Size, Sensitivity to Noise, Reasoning

Roberts, 2×2, Extremely High,
    It uses the smallest possible area. One bad pixel completely ruins the
    calculation.
Prewitt, 3×3, Medium,
    It averages across 3 pixels, which helps "dilute" the impact of one noisy
    pixel.
Sobel, 3×3, Low (Best),
    It uses weights (e.g., the center pixel is weighted by 2). This provides a
    "smoothing" effect built directly into the mask.

"""

"""
2) Canny
"""

# Basically that same functino above but with a gaussian blur at the start:


def canny():

    im_g = cvkit.filter.gaussian(im, 3, 1.4)

    G_x, G_y = cvkit.segment.gradient(im_g, cvkit.util.sobel_mask)

    orientation = cvkit.util.orientation(G_x, G_y)
    norme = cvkit.util.norme(G_x, G_y)

    thin = cvkit.segment.non_maximum_suppression(norme, orientation)
    hysteresis = cvkit.segment.hysteresis(thin)

    return hysteresis.astype(np.uint8)


cv2.imshow("Canny", canny())
cv2.waitKey(0)
