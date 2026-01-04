
from util import TruthTable
import numpy as np

verbose = False


class Perceptron:
    def __init__(self, weights, bias, activationMethod, inputNames=None):

        if inputNames is None:
            inputNames = [f"x{i}" for i in range(1, len(weights)+1)]

        if inputNames is not None and len(inputNames) != len(weights):
            raise ValueError("Each input has a weight in a Perceptron!")

        self.inputsLen = len(inputNames)
        self.inputNames = inputNames

        self.bias = bias
        self.weights = np.array(weights)
        self.activationMethod = activationMethod

    def linear_combination(self, inputs):
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def activation(self, inputs):
        return self.activationMethod(self.linear_combination(inputs))

    def fit(self, targets, iterations, learning_rate=0.001):

        booleanInputs = TruthTable.baseBools(self.inputsLen)
        losses = {}
        for i in range(self.inputsLen):
            losses[i] = []

        for _ in range(iterations):
            for i, (x, target) in enumerate(zip(booleanInputs, targets)):

                # Prediction: ŷ = w·x + b with x = [x1, x2, ..., xi]
                prediction = self.linear_combination(x)

                # Loss: L = (ŷ - y)²
                loss = (prediction - target)**2

                # Gradients:

                # ∂L/∂wi = 2(ŷ - y)·xi
                for index, (wi, xi) in enumerate(zip(self.weights, x)):
                    grad_wi = 2 * (prediction - target) * xi
                    self.weights[index] = wi - learning_rate * grad_wi  # (-) ???

                # ∂L/∂b = 2(ŷ - y)
                grad_bias = 2 * (prediction - target)
                self.bias = self.bias - learning_rate * grad_bias

    def printResult(self):
        print(end="\n")
        booleanInputs = TruthTable.baseBools(len(self.inputNames))
        outputs = [self.activation(row) for row in booleanInputs]
        TruthTable(self.inputNames, outputs).drawTable()
