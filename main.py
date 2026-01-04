
from mlkit import SingleNeuron, linear

singleNeuron = SingleNeuron(weights=[1.0, 1.0], bias=0.0, activationFunction=lambda x: 1.0 if x > 0.5 else 0.0)

singleNeuron.fit(targets=[0, 0, 0, 1], iterations=10000, learning_rate=0.001)
singleNeuron.printResult()
