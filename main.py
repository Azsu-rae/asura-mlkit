
from mlkit import Perceptron

perceptron = Perceptron(2, w=[1, 1], b=1, verb={"struct": True, "equ": False})

perceptron.fit([0, 0, 0, 1], eta=1, verb={
    "struct": True,
    "equ": False,
})

perceptron.display({"struct": True, "equ": False})
