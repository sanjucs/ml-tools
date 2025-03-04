# Activation function
Activation fuinction introduces non-linearity in neural network which helps it to learn complex patterns. Activation functions should be differentiable or should have minimum number of discontinuities. Commonly used activation functions are listed below. 

## Sigmoid function
Sigmoid function $σ(x)$ is defined as $σ(x) = {1 \over 1 + e^{-x}}$

* Output of sigmoid function lies between 0 and 1. Output approaches 0 as x tends to -inf and approaches 1 as x tends to inf.
* Differentiable at every point

Gradient computation:

$σ(x) = {1 \over 1 + e^{-x}}$

$σ^{'}(x) = {e^{-x} \over (1 + e^{-x})^{2}}$

$σ^{'}(x) = {1 \over 1 + e^{-x}} * {e^{-x} \over 1 + e^{-x}}$

$σ^{'}(x) = {1 \over 1 + e^{-x}} * {e^{-x} \over 1 + e^{-x}}$

$σ^{'}(x) = σ(x) * ( 1 - σ(x))$

<!-- 


2. Softmax function
3. Tanh function
4. ReLU
5. SiLU -->