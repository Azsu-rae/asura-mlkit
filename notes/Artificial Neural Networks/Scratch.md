
https://claude.ai/chat/bc003cb3-021a-4ff9-936c-09e535a6a575
https://www.youtube.com/watch?v=aircAruvnKk
https://bootcamp-2026.vercel.app/#/
# What is an Artificial Neural Network?

## Definition

In machine learning, A **Neuron** is a mathematical structure _vaguely_ inspired by the biological neuron, so let's start by looking at that first. from Wikipedia:

<div>
  <img src="images/Blausen_0657_MultipolarNeuron.png" align="left" width="40%" height="100%" style="margin-right: 20px; margin-bottom: 10px;">
  <p>
    “A <b>neuron</b> (American English), <b>neurone</b> (British English), or <b>nerve cell</b>, is a cell that is <b>excitable</b>, firing electric signals called <b>action potentials</b> across a <b>neural network</b> in the <b>nervous system</b>, mainly in the central nervous system, and helps to receive and conduct <b>impulses</b>. Neurons communicate with other cells via <b>synapses</b>, which are specialized connections that commonly use minute amounts of chemical neurotransmitters to pass the electric signal from the presynaptic neuron to the target cell through the synaptic gap.” TODO: add a reference
  </p>
</div>

while the anatomy of such a neuron is quite complexe, the mathematically inspired model of it is much more accessible. In machine learning, we abstract a few key behaviors:

- **Excitability**: The neuron's ability to be excited or, in machine learning terms, **activated**, in response to input signals
- **Signal propagation**: an activated neuron can transmit signals to other neurons in the network
- **Weighted connections**: some inputs influence the neuron more strongly than others

We usually visualize such an abstraction as a mathematical construct illustrated in the following diagram:
<div align="center" width="100%" height="100%"> <img src="tex/nn/neuron.svg"> </div>
We first need to define a **linear combination** of vectors (or vector components) as an expression where you multiply each term by a scalar and sum the results.

TODO: illustrate

We can therefore write the complete formula for the activation of a neuron, noted $z$, as the linear combination of the **input vector** $x = (x_1, x_2, ..., x_n)$  and a **weight vector** $w=(w_1, w_2, ..., w_n)$, plus a **constant offset** $b$:
$$ z = \sum_{i=0}^n w_i x_i + b = x_1 w_1 + x_2 w_2 + \dots + x_n w_n + b $$
TODO: motivate the bias

We usually want to chain such neurons to form neural networks networks as illustrated:

<div align="center" width="100%" height="100%"> <img src="tex/nn/ann.svg"> </div>

TODO: motivate the activation function

## The Perceptron

Let us start by looking at the simplest form of a neural network developed in the 1950s (although neural networks themselves were first invented in 1943).

The **Perceptron** is a **supervised learning** algorithm for **binary classification**.

TODO: definition and truth table learning
TODO: XOR, linear separability, and MLPs
# Gradient Descent

Let's consider a simple neural network with only 4 neurons:

<div align="center" width="100%" height="100%"> <img src="tex/nn/gradient_descent_example.svg"> </div>

TODO: switch to a linear regression example
POTENTIALLY: convex, and Normal Equation closed-form solution, linearity

...