# Activation function
Activation functions introduce non-linearities in neural networks, which helps them learn complex patterns. Activation functions should be differentiable or should have minimum number of discontinuities. Commonly used activation functions are listed below. 

## Sigmoid function
The sigmoid function $σ(x)$ is defined as $σ(x) = {1 \over 1 + e^{-x}}$

* Output of sigmoid function lies between 0 and 1. Output approaches 0 as x tends to -inf and approaches 1 as x tends to inf.
* Differentiable at every point

Gradient computation:

$σ(x) = {1 \over 1 + e^{-x}}$

$σ^{'}(x) = {e^{-x} \over (1 + e^{-x})^{2}}$

$σ^{'}(x) = {1 \over 1 + e^{-x}} * {e^{-x} \over 1 + e^{-x}}$

$σ^{'}(x) = {1 \over 1 + e^{-x}} * {e^{-x} \over 1 + e^{-x}}$

$σ^{'}(x) = σ(x) * ( 1 - σ(x))$

## Tanh function
The tanh is a hyperbolic function and is defined as $tanh(x) = {sinh(x) \over conhx(x)}$

$tanh(x) = {sinh(x) \over coshx(x)}$

$tanh(x) = {{e^{x}-e^{-x}} \over {{e^{x} + e^{-x}}}}$

$ {{d\over dx} tanh(x)} = {{e^{x}-e^{-x}} \over {{e^{x} + e^{-x}}}}$


## The softmax function
$softmax(x_i) = {e^{-x_i} \over \sum_{k}e^{-x_k}}$

## ReLU

Rectified Linear Unit is defined as $ReLU(x) = max(0, x)$

## SiLU
Sigmod Linear Unit is defined as $SiLU(x) = x * sigmoid(x)$