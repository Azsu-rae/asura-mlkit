
The classic perceptron uses a hard threshold, usually the *Heaviside* step function:
$ accent(y, hat) = f(z) = cases(
  1 "if" z >= 0,
  0 "if" z < 0,
) $

If using $\{−1, +1\}$ labels instead:

#let sign = math.op("sign")
$ accent(y, hat) = sign(z) $

For a single training example, the weight update rule is:

$ w_i <- w_i + eta e x_i $

and the bias update is:

$ b <- b + eta e $

with

$ e = y − accent(y, hat) $

notice there are only 2 possible values for $e$:

$
  e = cases(
    1 "if" (y, accent(y, hat)) = (1, 0),
    -1 "if" (y, accent(y, hat)) = (0, 1),
  )
$

what do either of these cases mean? Well, if it's 1, that means that output was negative when it was supposed to be positive. And if it
was -1 that means that the output was positive when it was supposed to be negative. So what are we doing multiplying this $e$ by the input
value to 'train'? Let's look at how we compute our output in more detail:

$ z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b $

we are multiplying each weight by the _value of the input $x_i$_. So if the input is positive and we want the output to be positive then
we want $x_i w_i$ to be positive! So the weight should also be positive! And so we need to converge to that positive weight if it happens
to be negative which is why we add $x_i e$ to the weight. If the result was positive
