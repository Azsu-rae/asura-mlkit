
from util import TruthTable, verbose
import numpy as np


class Perceptron:
    def __init__(self, nb_inputs, w=None, b=None):

        self.inputNames = [f"x{i}" for i in range(1, nb_inputs+1)]
        self.nb_inputs = nb_inputs

        self.w = np.array(w)
        if w is None:
            self.w = np.zeros(self.nb_inputs)

        self.b = b
        if b is None:
            self.b = 0

        verbose.initzip("weights", w, prefix="w")
        verbose.init("bias", "b", b)

    def fit(self, y, epochs=10000, eta=0.01):

        input_examples = TruthTable.baseBools(self.nb_inputs)
        y = np.array(y)

        verbose.initzip(
                "input examples",
                zip(input_examples, y),
                len(y),
                prefix="Example ")

        for epoch in range(epochs):
            for example_index, x in enumerate(input_examples):

                y_hat = self.predict(x)
                e = y[example_index] - y_hat
                for i in range(self.nb_inputs):
                    self.w[i] += eta * e * x[i]

                self.b += eta * e

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        y_hat = 1 if z >= 0 else 0
        return y_hat

    def displayTruthTable(self):
        booleanInputs = TruthTable.baseBools(self.nb_inputs)
        outputs = [self.predict(row) for row in booleanInputs]
        TruthTable(self.inputNames, outputs).drawTable()

    def verbose(self, example_index, y, epoch):
        print("{}")
