### Sigmoïde

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$
- Used for multi-label
- Interesting derivative:

$$
\sigma'(x) = \sigma(x) (1-\sigma(x))
$$
### Softmax

For a vector $z$ of length $K$:

$$
Softmax(z_i) = \frac{e^{z_i}}{\Sigma_{j=1}^{K} e^{z_j}}
$$
- Used for multi-class
