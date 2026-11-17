
from cvkit.util import convolution

import numpy as np
import cvkit

# ----------------- moyenneur -----------------


def convolve_average(im, i, j, ker):
    pad = ker // 2
    avr = im[i-pad:i+pad+1, j-pad:j+pad+1].sum() / ker**2
    return np.uint8(avr)


def mean(im, ker):
    return convolution(im, ker, lambda i, j: convolve_average(im, i, j, ker))

# ----------------- median -----------------


def convolve_middle(im, i, j, ker):
    pad = ker // 2
    return np.median(im[i-pad:i+pad+1, j-pad:j+pad+1])


def median(im, ker):
    return convolution(im, ker, lambda i, j: convolve_middle(im, i, j, ker))

# ----------------- gaussian -----------------


def convolve_gaussian(im, i, j, ker, sig):

    pad = ker // 2
    region = im[i-pad:i+pad+1, j-pad:j+pad+1]
    gaussian_ker = cvkit.util.gaussian_kernel(ker, sig)

    return np.uint8((region * gaussian_ker).sum())


def gaussian(im, ker, sigma):
    return convolution(im, ker, lambda i, j: convolve_gaussian(im, i, j, ker, sigma))

# ----------------- bilateral -----------------


def bi_weighted_convolution(im, i, j, ker, sigma_s, sigma_r):

    sum_weights = 0.0
    sum_weighted_values = 0.0

    pad = ker // 2
    for k in range(-pad, pad+1):
        for l in range(-pad, pad+1):

            spatial_dist = np.sqrt(k*k + l*l)
            spatial_weight = np.exp(-spatial_dist**2 / (2 * sigma_s**2))
            # or spatial_weight = np.exp(-(k*k + l*l) / (2 * sigma_s**2))

            color_dist = abs(im[i, j].astype(int)-im[i+k, j+l].astype(int))
            color_weight = np.exp(-color_dist**2 / (2 * sigma_r**2))

            weight = spatial_weight * color_weight

            sum_weights += weight
            sum_weighted_values += weight * im[i+k, j+l]

    resultat = (sum_weighted_values // sum_weights)

    return np.uint8(resultat)


def bilateral(im, ker, sigma_s, sigma_r):
    return convolution(im, ker, lambda i, j: bi_weighted_convolution(im, i, j, ker, sigma_s, sigma_r))

# ----------------- nagao -----------------


def region_pixels(im, reg, i, j):

    pixels = []
    for coordoner in reg:
        k, l = coordoner
        pixels.append(im[i+k, j+l])

    return np.array(pixels, dtype=np.float32)


def convolve_nagao_region(im, i, j, ker):

    regions = cvkit.util.nagao_regions(ker)
    good_region = 0
    min_var = float("inf")

    for k, r in enumerate(regions):
        var = np.var(region_pixels(im, r, i, j))
        if var < min_var:
            min_var = var
            good_region = k

    return np.uint8(region_pixels(im, regions[good_region], i, j).sum() / len(regions[good_region]))


def nagao(im, ker=5):
    return convolution(im, ker, lambda i, j: convolve_nagao_region(im, i, j, ker))
