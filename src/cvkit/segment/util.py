
import numpy as np


def norme(G_x, G_y):
    return np.sqrt(G_x.astype(np.float32)**2, G_y.astype(np.float32)**2)


def orientation(G_x, G_y):
    return np.arctan2(G_x, G_y)


sobel_mask = {
    "G_x": [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], "G_y": [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ], "ker": 3
}

robert_mask = {
    "G_x": [
        [0, 1],
        [-1, 0],
    ], "G_y": [
        [1, 0],
        [0, -1],
    ], "ker": 2
}

prewitt_mask = {
    "G_x": [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], "G_y": [
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ], "ker": 3
}
