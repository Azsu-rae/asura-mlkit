
from util import TruthTable
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary(x):
    return True if x > 0 else False


class Perceptron:
    def __init__(self, inputNames, weights, bias, activationMethod):

        if len(inputNames) != len(weights):
            raise ValueError("Each input has a weight in a Perceptron!")

        self.inputNames = inputNames
        self.weights = weights
        self.bias = bias
        self.activationMethod = activationMethod

    def activation(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activationMethod(total)

    def printResult(self):
        print()
        TruthTable(self.inputNames, [self.activation(row) for row in TruthTable.baseBools(len(self.inputNames))]).drawTable()
