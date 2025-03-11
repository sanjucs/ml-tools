# Activation functions
Activation functions introduce non-linearities in neural networks, which helps them learn complex patterns. Activation functions should be differentiable or should have minimum number of discontinuities. Commonly used activation functions are listed below. 

## Sigmoid
The sigmoid function $σ(x)$ is defined as 

$$σ(x) = {1 \over 1 + e^{-x}}$$

* Output of sigmoid function lies between 0 and 1. Output approaches 0 as x tends to -inf and approaches 1 as x tends to inf.
* Differentiable at every point

Gradient computation:

$$σ(x) = {1 \over 1 + e^{-x}}$$

$$σ\prime(x) = {e^{-x} \over (1 + e^{-x})^{2}}$$

$$σ\prime(x) = {1 \over 1 + e^{-x}} \cdot {e^{-x} \over 1 + e^{-x}}$$

$$σ\prime(x) = {1 \over 1 + e^{-x}} \cdot {e^{-x} \over 1 + e^{-x}}$$

$$σ\prime(x) = σ(x) \cdot ( 1 - σ(x))$$

## Tanh
The tanh function is a hyperbolic function and is defined as

$$tanh(x) = {sinh(x) \over conhx(x)}$$

Gradient computation:

$$tanh(x) = {sinh(x) \over coshx(x)}$$

$$tanh(x) = {{e^{x}-e^{-x}} \over {{e^{x} + e^{-x}}}}$$

$${{d \over dx}} tanh(x) = {{d \over dx}} {{e^{x}-e^{-x}} \over {e^{x} + e^{-x}}}$$

$${{d \over dx}} tanh(x) = {{(e^{x}+e^{-x}) \cdot (e^{x}+e^{-x}) - (e^{x}-e^{-x}) \cdot (e^{x}-e^{-x})} \over {(e^{x} + e^{-x})^2}}$$

$${{d \over dx}} tanh(x) = {{(e^{x}+e^{-x})^2 - (e^{x}-e^{-x})^2} \over {(e^{x} + e^{-x})^2}}$$

$${{d \over dx}} tanh(x) = 1 - {\left({e^{x}-e^{-x}} \over {e^{x} + e^{-x}}\right)^2}$$

$${{d \over dx}} tanh(x) = 1 - tanh^2(x)$$

$${{d \over dx}} tanh(x) = sech^2(x)$$


## Softmax
The softmax function is defined as

$$softmax(x_i) = {e^{x_i} \over \sum_{k}e^{x_k}}$$

The optimized softmax calculation suggested in the flash attention paper can be found [here](/notes/dl/modules/softmax.py)

## ReLU

Rectified Linear Unit is defined as 

$$ReLU(x) = max(0, x)$$

## SiLU
Sigmod Linear Unit is defined as

$$SiLU(x) = x \cdot sigmoid(x)$$
