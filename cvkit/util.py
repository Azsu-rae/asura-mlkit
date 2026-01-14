
from cvkit.segment.util import *
from cvkit.filter.util import *

import matplotlib.pyplot as plt
import numpy as np


def normalize(values, scale):
    diff = values.max() - values.min()
    if diff == 0:
        return np.zeros_like(values)
    return (values - values.min()) / diff * scale


def pad(ker):
    if ker % 2 == 0:
        pad_s, pad_e = 0, ker-1
    else:
        pad_s, pad_e = ker // 2, ker // 2
    return pad_s, pad_e


def convolution(im, ker, convolve, dtype=np.uint8):

    pad_s, pad_e = pad(ker)
    h, w = im.shape[:2]

    output = im[pad_s:-pad_e, pad_s:-pad_e].copy().astype(dtype)
    for i in range(pad_s, h-pad_e):
        for j in range(pad_s, w-pad_e):
            output[i-pad_s][j-pad_e] = convolve(i, j)

    return output


def mse(im1, im2):

    if im1.shape != im2.shape:
        raise ValueError("MSE is calculated with images of the same shape!")

    return np.mean((im1.astype(np.float64) - im2.astype(np.float64)) ** 2)


def psnr(im, imf):

    MSE = mse(im, imf)
    if mse == 0:
        return float('inf')

    R = 255.0
    return 10 * np.log10((R ** 2) / MSE)

def histogram(im):

    h, w, channels = im.shape
