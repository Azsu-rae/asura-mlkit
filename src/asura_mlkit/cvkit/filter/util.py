
import numpy as np
import mathkit


def binomial_coeffs(n):

    coeffs = []
    for k in range(n+1):
        coeffs.append(mathkit.C(n, k))

    return np.array(coeffs)


def binomial_gaussian_kernel(n):
    coeffs = np.array(binomial_coeffs(n))
    return np.outer(coeffs, coeffs) / binomial_coeffs(n).sum()**2


def gaussian_kernel(ker, sig):

    pad = ker // 2

    ax = np.arange(-pad, pad+1)
    xx, yy = np.meshgrid(ax, ax)
    expr = -(xx**2 + yy**2) / (2 * sig**2)
    gauss_ker = np.exp(expr)

    return gauss_ker / gauss_ker.sum()


def nagao_regions(ker):

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

    return [square,
            up, down, left, right,
            south_east, south_west, north_east, north_west]
