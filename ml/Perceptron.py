
from ml.TruthTable import TruthTable

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

    def backpropagation(self):
        a =True
        print(a)
        # TODO MAKE SOMETHING


