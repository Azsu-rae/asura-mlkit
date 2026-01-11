
from mlkit import Perceptron

perceptron = Perceptron(2)

perceptron.fit([0, 0, 0, 1], eta=0.1, verb={
    "struct": True,
    "equ": True,
})

perceptron.display({"struct": True, "equ": False})
