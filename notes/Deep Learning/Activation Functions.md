### Sigmoïde
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

- Used for multi-label
### Softmax

For a vector $z$ of length $K$:
$$
Softmax(z_i) = \frac{e^{z_i}}{\Sigma_{j=1}^{K} e^{z_j}}
$$
- Used for multi-class