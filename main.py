
import numpy as np
from mlkit import Perceptron
from mlkit import linear

inputs = np.array([1, 2, 3])

w = np.array([[0, 0, 0], [0, 0, 0]])

b = np.array([0, 0])

Perceptron(['x1', 'x2'], [1.0, 2.0], 2.0, linear).printResult()
