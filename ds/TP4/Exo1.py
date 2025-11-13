
import numpy as np

T = np.array([
    [22, 25, 20],
    [24, 27, 21],
    [19, 20, 22],
    [25, 29, 28],
    [26, 30, 27],
    [21, 21, 23],
    [20, 26, 25],
])

print(np.max(T, axis=0)) # columnm
print(np.max(T, axis=1)) # line
print(np.max(np.max(T, axis=0))) # overall
