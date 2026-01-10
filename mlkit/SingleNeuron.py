
from util import TruthTable

import numpy as np

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt  # noqa: E402


class SingleNeuron:
    def __init__(self, weights, bias, activationFunction, inputNames=None):

        if inputNames is None:
            inputNames = [f"x{i}" for i in range(1, len(weights)+1)]

        if len(inputNames) != len(weights):
            raise ValueError("Each input has a weight in a Perceptron!")

        self.inputsLen = len(inputNames)
        self.inputNames = inputNames

        self.bias = bias
        self.weights = np.array(weights)
        self.activationFunction = activationFunction

    def linear_combination(self, inputs):
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def predict(self, inputs):
        return self.activationFunction(self.linear_combination(inputs))

    def fit(self, targets, epochs, learning_rate=0.001):

        booleanInputs = TruthTable.baseBools(self.inputsLen)
        loss = []

        for it in range(epochs):
            loss.append(0)
            for i, (x, target) in enumerate(zip(booleanInputs, targets)):

                # Prediction: ŷ = w·x + b with x = [x1, x2, ..., xi]
                prediction = self.linear_combination(x)

                # Loss: L = (ŷ - y)²
                loss[it] += (prediction - target)**2

                # Gradients:

                # ∂L/∂wi = 2(ŷ - y)·xi
                for index, (wi, xi) in enumerate(zip(self.weights, x)):
                    grad_wi = 2 * (prediction - target) * xi
                    self.weights[index] = wi - learning_rate * grad_wi  # (-) ???

                # ∂L/∂b = 2(ŷ - y)
                grad_bias = 2 * (prediction - target)
                self.bias = self.bias - learning_rate * grad_bias

            loss[it] /= len(targets)

        self.loss = loss

    def printResult(self):
        print(end="\n")
        booleanInputs = TruthTable.baseBools(len(self.inputNames))
        outputs = [self.predict(row) for row in booleanInputs]
        TruthTable(self.inputNames, outputs).drawTable()

        print(end="\n")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")
        if hasattr(self, "loss"):
            # self.loss = np.array(self.loss, dtype=float)
            plt.semilogy(range(len(self.loss)), self.loss)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            # plt.grid(True)
            # plt.savefig("loss.png")
            plt.show()
