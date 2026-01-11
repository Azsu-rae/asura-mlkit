
from util.verbose import initzip, init, start, step, equation, state, writeIf, pad
from util import TruthTable

import numpy as np


class Perceptron:
    def __init__(self, nb_inputs, w=None, b=None, verb=None):

        if verb is None:
            verb = {
                "struct": False,
                "equ": False,
            }

        self.inputNames = [f"x{i}" for i in range(1, nb_inputs+1)]
        self.nb_inputs = nb_inputs

        self.w = np.array(w)
        if w is None:
            self.w = np.zeros(self.nb_inputs)

        self.b = b
        if b is None:
            self.b = 0

        initzip("weights", self.w, prefix="w", verb=verb)
        init("bias", "b", self.b, verb=verb)
        self.epoch = 0
        self.display(verb=verb)

    def fit(self, y, epochs=10000, eta=0.01, verb=None):

        if verb is None:
            verb = {
                "struct": False,
                "equ": False,
            }

        input_examples = TruthTable.baseBools(self.nb_inputs)
        y = np.array(y)

        start("Iterations: ", verb=verb["struct"])
        for epoch in range(epochs):
            step("Epoch", epoch+1, verb=verb["struct"])
            error = True
            for example_index, x in enumerate(input_examples):

                step("Example", example_index+1, verb=verb["struct"])
                pad(verb=verb["struct"])
                state("w", self.w, verb=verb["struct"])
                state("b", round(self.b, 2), verb=verb["struct"])

                y_hat = self.predict(x, verb=verb["equ"])
                e = y[example_index] - y_hat
                equation(
                        "e = (y - y_hat)",
                        f"e = ({y[example_index]} - {y_hat})",
                        e, verb=verb["equ"])

                error = error and (e == 0)
                writeIf(
                        e == 0,
                        "output is correct! No need to update the parameters.",
                        f"Incorrect output ({e})! Updating parameters...",
                        verb=verb["struct"])

                for i in range(self.nb_inputs):

                    delta_wi = eta * e * x[i]
                    equation(
                            f"delta_w{i} = eta * x{i} * e",
                            f"delta_w{i} = {eta} * {x[i]} * {e}",
                            delta_wi, e != 0,
                            verb=verb["equ"])

                    equation(
                            f"w{i} = w{i} + delta_w{i}",
                            f"w{i} = {round(self.w[i], 2)} + {delta_wi}",
                            round(self.w[i]+delta_wi, 2), e != 0,
                            verb=verb["equ"])
                    self.w[i] += delta_wi

                equation(
                        "b = b + eta * e",
                        f"b = {round(self.b, 2)} + {eta} * {e}",
                        round(self.b + eta*e, 2), e != 0,
                        verb=verb["equ"])
                self.b += eta * e
                self.epoch = epoch
                self.display(verb={"struct": verb["struct"], "equ": False})

            if (self.outputs(False) == y).all():
                return

        self.epoch = None

    def predict(self, x, verb=True):

        z = np.dot(self.w, x) + self.b
        y_hat = 1 if z >= 0 else 0

        pad(verb=verb)
        state("for x", x, verb=verb)
        equation(
                "z = ∑w.x + b",
                f"z = {round(np.dot(self.w, x), 2)} + {round(self.b, 2)}",
                round(z, 2), verb=verb)
        equation(
                "y_hat = 1 if z >= 0 else 0",
                f"y_hat = 1 if {round(z, 2)} >= 0 else 0",
                y_hat, verb=verb)

        return y_hat

    def display(self, verb=None):

        if verb is None or verb["struct"] is False:
            return

        if self.epoch is None:
            start("Reached the max epochs with no result!", verb=verb["struct"])

        pad(verb=verb["struct"])
        state("epoch", self.epoch, verb=verb["struct"])
        state("w", self.w, verb=verb["struct"])
        state("b", round(self.b, 2), verb=verb["struct"])

        outputs = self.outputs(verb["equ"])
        TruthTable(self.inputNames, outputs).drawTable()

    def outputs(self, verb):
        booleanInputs = TruthTable.baseBools(self.nb_inputs)
        return [self.predict(row, verb) for row in booleanInputs]
