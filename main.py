
from mlkit import Perceptron


perceptron = Perceptron(2, [1, 0.5], -0.5)

perceptron.fit([0, 0, 0, 1])
# perceptron.displayTruthTable()
