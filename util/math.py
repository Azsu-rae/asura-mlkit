
import numpy as np


def fac(n):

    if n <= 0:
        return 1

    return n * fac(n-1)


def C(n, k):
    return fac(n) // (fac(k) * fac(n-k))


def normal_pdf(x, mean, std):

    # e^(-(x-μ)²/(2σ²))
    exponential_kernel = np.exp(- (x - mean)**2 / (2 * std**2))

    # 1 / σ * √(2π)
    normalization_constant = 1 / (std * np.sqrt(2*np.pi))

    # (1 / (σ * √(2π))) * e^(-(x-μ)²/(2σ²))
    return normalization_constant * exponential_kernel
