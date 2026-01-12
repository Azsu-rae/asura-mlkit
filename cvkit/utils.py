
import matplotlib.pyplot as plt
import numpy as np
import mathkit as math

def mse(im, imf):

    if im.shape != imf.shape:
        raise ValueError("MSE is calculated with images of the same size")

    return np.mean((im.astype(np.float64) - imf.astype(np.float64)) ** 2)


def psnr(im, imf):

    MSE = mse(im, imf)
    if mse == 0:
        return float('inf')

    R = 255.0
    return 10 * np.log10((R ** 2) / MSE)


def binomial_coeffs(n):

    coeffs = []
    for k in range(n+1):
        coeffs.append(math.C(n, k))

    return np.array(coeffs)


def binomial_gaussian_kernel(n):
    coeffs = np.array(binomial_coeffs(n))
    return np.outer(coeffs, coeffs) / binomial_coeffs(n).sum()**2


def gaussian_kernel(ker, sig):

    pad = ker // 2

    ax = np.arange(-pad, pad+1)
    xx, yy = np.meshgrid(ax, ax)
    expr = -(xx**2 + yy**2) / 2 * sig**2
    gauss_ker = np.exp(expr)

    return gauss_ker / gauss_ker.sum()


def nagao_regions():

    square = [
        (-1, -1), (-1, 0), (-1, +1),
        (0, -1), (0, 0), (0, +1),
        (+1, -1), (+1, 0), (+1, +1),
    ]

    up = [
        (0, 0), (-1, 0), (-2, 0),
        (-1, -1), (-2, -1),
        (-1, +1), (-2, +1),
    ]

    down = [
        (0, 0), (+1, 0), (+2, 0),
        (+1, -1), (+2, -1),
        (+1, +1), (+2, +1),
    ]

    left = [
        (0, 0), (0, -1), (0, -2),
        (+1, -1), (+1, -2),
        (-1, -1), (-1, -2),
    ]

    right = [
        (0, 0), (0, +1), (0, +2),
        (+1, +1), (+1, +2),
        (-1, +1), (-1, +2),
    ]

    south_west = [
        (0, 0), (0, -1), (-1, 0),
        (-1, -1), (-2, -2),
        (-2, -1), (-1, -2),
    ]

    south_east = [
        (0, 0), (0, +1), (-1, 0),
        (-1, +1), (-2, +2),
        (-2, +1), (-1, +2),
    ]

    north_east = [
        (0, 0), (0, +1), (+1, 0),
        (+1, +1), (+2, +2),
        (+2, +1), (+1, +2),
    ]

    north_west = [
        (0, 0), (+1, 0), (-1, 0),
        (-1, +1), (-2, +2),
        (-2, +1), (+2, -1),
    ]

    return [square, up, down, left, right, south_east, south_west, north_east, north_west]


def histo_gray(im):

    h, w = im.shape[:2]

    histo = np.zeros(256)
    for i in range(h):
        for j in range(w):
            niveaux_gray = im[i, j]
            histo[niveaux_gray] += 1

    return histo


def histo_color(im):

    h, w = im.shape[:2]

    print(im[0, 0])
    histo = np.zeros((3, 255))
    for i in range(h):
        for j in range(w):

            blue = im[i, j][0]
            green = im[i, j][1]
            red = im[i, j][2]

            histo[0][blue] += 1
            histo[1][green] += 1
            histo[2][red] += 1

    return histo


def print_gray_histo(histo):
    print("niveaux_gray   occurences")
    for niveaux_gray, occurences in enumerate(histo):
        print(f"     {niveaux_gray}             {occurences}")


def normalization(im, threshold=0, Y_min=0, Y_max=255):

    histo = histo_gray(im)

    X_min = -1
    X_max = -1
    for i in range(len(histo)):
        if histo[i] > threshold and X_min == -1:
            X_min = i
        if histo[255-i] > threshold and X_max == -1:
            X_max = 255-i
        if X_min != -1 and X_max != -1:
            break

    h, w = im.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            ratio = ((im[i, j] - X_min) / (X_max - X_min))
            out[i, j] = (255 * ratio).astype(np.uint8)

    return out


def equalization(im):

    histo = histo_gray(im)
    cdf = histo.cumsum()

    no_pxls = histo.sum()
    h, w = im.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            level = im[i, j]
            out[i, j] = ((cdf[level] - cdf.min()) / (no_pxls - cdf.min())) * 255

    return out


def display_gray_histo(histo):

    x = range(0, 256)
    y = histo

    plt.plot(x, y)
    plt.xlabel("gray levels")
    plt.ylabel("occurences")
    plt.title("histogram of image")
    plt.show()
