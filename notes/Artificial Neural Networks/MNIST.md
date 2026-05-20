
<div align="center" width="100%" height="100%">
	<img src="tex/nn/mnist.svg">
</div>

if we were to consider the activation of the $k_{th}$ neuron in the layer $l$ as illustrated:
<div align="center" width="100%" height="100%">
	<img src="tex/nn/kth_neuron.svg">
</div>

the corresponding equation is,

$$
z^{(l)}_k = \Sigma_{i=1}^{n} w_{k, i} a^{(l-1)}_i + b^{(l)}_k
$$

We'll be using the sigmoïde ($\sigma$) activation function and so,

$$
a^{(l)}_k = \sigma{(z^{(l)}_k)}
$$

Gradient descent uses the derivative of the cost function $C$ which is defined as

$$
C = \Sigma_k\frac{1}{2}(a^{(l)}_k - y_k^{(l)})^2
$$

where $k$ means the $k_{th}$ neuron in the output layer. $y_k^{(l)}$ is the desired output. That $\frac{1}{2}$ is there to cancel that exponent of $2$ when deriving. We take the sum of the (squared) differences because any given activation $a^{(l-1)}$ will affect all the neurons in the output layer. We don't really have any way to know where the error came from so we just sum all of them.

For the sake of simplifying the computations and notations, we will omit the $k$ subscript and $(l)$s for all subsequent steps while trying leaving no room for confusion.

To apply the gradient descent algorithm, we need to take the derivative of the loss $L$ relative to the weights and bias of that loss function.  the derivative relative to a given weight $w_{k,i}$  is defined as,

$$
\begin{aligned}
\frac{\partial L}{\partial w_{k,i}}
&=  \frac{1}{2} \frac{\partial (a^{(l)} - y)^2}{\partial w_{k,i}} \\ \\
&= \frac{1}{2} \times 2 (a^{(l)}-y)\frac{\partial a^{(l)}}{\partial w_{k,i}}
&\text{ (apply the chain rule)}\\ \\
&= (a^{(l)}-y)\frac{\partial \sigma(z^{(l)}_k)}{\partial w_{k,i}}
&\text{ (plug in $a^{(l)}$)}\\ \\
\end{aligned}
$$

Of course we have,

$$
\frac{\partial\sigma(z^{(l)}_k)}{\partial w_{k,i}}
=\frac{\partial\sigma(\Sigma_{j=1}^{n} w_{k,j} a^{(l-1)}_j + b)}{\partial w_{k,i}}
=\frac{
	\partial\sigma(
		w_{k,1} a^{(l-1)}_1 + \dots
		 + w_{k,i} a^{(l-1)}_i + \dots
		 + w_{k,n} a^{(l-1)}_n + b
	 )
 }{\partial w_{k,i}}
$$

applying the chain rule, all the $w_{k,j}\ a_j$ elements where $j\neq i$  are constants and have a derivative $=0$. The only remaining element is where $j=i$ which is $w_{k,i}\ a_i^{(l-1)}$ and derives to $a_i^{(l-1)}$. Thus,

$$
\frac{\partial\sigma(z^{(l)}_k)}{\partial w_{k,i}}
= a_i^{(l-1)} \sigma'(z_k^{(l)})
$$

note that the same procedure when considering the derivative relative to $b_k$ yields:

$$
\frac{\partial \sigma(z_k^{(l)})}{b_{k}^{(l)}}=\sigma'(z^{(k)}_k)
\ \ \ \ \ \ \ \ \ \
\text{(since the derivative of $b$ is 1)}
$$

plugging this result back in the cost function derivative we get,

$$
\frac{\partial L}{\partial w_{k,i}^{(l)}}
= (a_k^{(l)}-y)\,a_i^{(l-1)}\,\sigma'(z^{(l)})
$$

and for $b_k^{(l)}$,

$$
\frac{\partial L}{\partial b_k^{(l)}}
= (a_k^{(l)}-y)\ \sigma'(z^{(l)})
$$

the sigmoïde has an interesting property:

$$
\sigma'(x) = \sigma(x) (1-\sigma(x))
$$

so using that in our formula we get,

$$
\frac{\partial L_k}{\partial w_{k,i}^{(l)}}
= (a_k^{(l)} - y)\ a_i^{(l-1)}\ \sigma(z_k^{(l)})(1-\sigma(z_k^{(l)}))
$$

note that the only element(?) in that derivative that is specific to the $i_{th}$ weight $w_{k,i}$ is $a_i^{(l-1)}$ so we can define an **output delta** as:

$$
\delta_k^{(l)}=(a_k^{(l)}-y)\sigma'(z_k^{(l)})
$$

Note that this corresponds to the derivative relative to $b_k^{(l)}$, and so this simplifies our final formulas to:

$$
\frac{\partial L_k}{\partial w^{(l)}_{k,i}}=a_i^{(l-1)} \delta^{(l)}_k
$$

and,

$$
\frac{\partial L_k}{\partial b_k^{(l)}}=\delta^{(l)}_k
$$

this makes clearer the relation to the $i_{th}$ weight and abstracts away the activation function $\sigma$ if we ever wish to change it or re-use this formula on another network.

We define the weight and bias update mechanisms as,

$$
\begin{align}
& w_{k,i}^{(l)} \leftarrow
w_{k,i}^{(l)} - \eta \frac{\partial L_k}{\partial w_{k,i}^{(l)}} \\
& b_k^{(l)} \leftarrow
b_k^{(l)} - \eta \frac{\partial L_k}{\partial b_k^{(l)}}
\end{align}
$$

A more concise notation:

$$
\begin{align}
& \Delta w^{(l)}_{k,i} = -\eta\ \delta_k^{(l)} a_i^{(l-1)} \\ \\
& \Delta b_k^{(l)} = -\eta\ \delta_k^{(l)}
\end{align}
$$

And that is the weight update formula of all the last layer's weights. But how about the updates across layers?
<div align="center" width="100%" height="100%">
	<img src="tex/nn/cross_layer.svg">
</div>
we'll try not to forget about the $b_i^{(l-1)}$. We didn't put it in the diagram because the space is kind of cramped.

How would we go about computing the weight update of any $w_{i,3}$? Well, we'll just take the error of all of the output neurons, sum them, and propagate them backward. That is what is called in machine learning **backpropagation**.

$$
w_{i,3} \leftarrow w_{i,3}-\eta\ \delta^{(l-1)}_i\ a_3^{(l-2)}
$$

$$
\Sigma_{k=1}^{10}
$$
