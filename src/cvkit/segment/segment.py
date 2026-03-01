
from cvkit.util import convolution

import numpy as np
import cvkit.util
import cvkit


def convolve_mask(im, i, j, mask, ker):
    pad_s, pad_e = cvkit.util.pad(ker)
    slice = im[i-pad_s:i+pad_e+1, j-pad_s:j+pad_e+1]
    return (slice.astype(np.int16) * mask).sum()


def gradient(im, mask):

    G_x = convolution(im, mask["ker"],
                      lambda i, j:
                          convolve_mask(im, i, j, mask["G_x"], mask["ker"]),
                      dtype=np.int16)

    G_y = convolution(im, mask["ker"],
                      lambda i, j:
                          convolve_mask(im, i, j, mask["G_y"], mask["ker"]),
                      dtype=np.int16)

    return G_x, G_y


def suppress_if_non_local_maximum(norme, i, j, orientation):

    angle = np.rad2deg(orientation[i, j]) % 180
    if angle <= 22.5 or angle > 157.5:
        # 0° case
        if norme[i, j-1] > norme[i, j] or norme[i, j+1] > norme[i, j]:
            return 0
        return norme[i, j]
    elif 22.5 < angle and angle <= 67.5:
        # 45° case
        if norme[i-1, j+1] > norme[i, j] or norme[i+1, j-1] > norme[i, j]:
            return 0
        return norme[i, j]
    elif 67.5 < angle and angle <= 112.5:
        # 90° case
        if norme[i-1, j] > norme[i, j] or norme[i+1, j] > norme[i, j]:
            return 0
        return norme[i, j]
    elif angle <= 157.5:
        # 135° case
        if norme[i-1, j-1] > norme[i, j] or norme[i+1, j+1] > norme[i, j]:
            return 0
        return norme[i, j]

    print(f"We are returning NONE BABY {orientation[i, j]} became {angle}")
    return None


def non_maximum_suppression(norme, orientation):
    return convolution(norme, 3,
                       lambda i, j:
                           suppress_if_non_local_maximum(norme, i, j, orientation),
                       dtype=np.float32)


def threshhold(thin, i, j, Thaut, Tbas):

    if thin[i, j] >= Thaut:
        return 255
    elif thin[i, j] >= Tbas and thin[i, j] < Thaut:
        if (thin[i-1:i+2, j-1:j+2] > Thaut).any():
            return 255
        else:
            return 0
    elif thin[i, j] < Tbas:
        return 0


def hysteresis(thin, Thaut=70, Tbas=30):
    return convolution(thin, 3,
                       lambda i, j:
                           threshhold(thin, i, j, Thaut, Tbas),
                       dtype=np.float32)
