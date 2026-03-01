
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary(x):
    return True if x > 0 else False


def linear(x):
    return x
