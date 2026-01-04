
from mlkit import Perceptron, linear

perceptron = Perceptron(weights=[1.0, 1.0], bias=0.0, activationMethod=linear)

perceptron.fit(targets=[0, 1, 1, 0], iterations=10000, learning_rate=0.001)
perceptron.printResult()
