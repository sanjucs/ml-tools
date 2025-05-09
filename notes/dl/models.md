# Evolution of AI models

## FC
## CNN
## RNN
Recurrent neural networks (RNNs) are designed to handle sequential data by processing one timestep at a time, after capturing information from the previous timestep into a single vector called the hidden state. By the sequential nature of how the hidden state is processed, it can hold information from any of the previous timesteps.


At time $t$ a RNN unit receives the current input $x_t$ and previous hidden state $h_{t-1}$ results new hidden state $h_t$ and output $y_t$.

$$h_t = \Phi(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = \Phi(W_{hy} h_t + b_y)$$

where

$x_t$ :- input at time $t$

$h_t$ :- hidden state at time $t$

$y_t$ :- output predicited at time $t$

$W$ :- Trainable weights

RNN needs to consider current input and previous hidden state for generating next output. Thus computation scales only linearly with sequence length. However there are 2 significant issues with RNN.
* Data must be processed sequentially, which drastically reduces the performance.
* Longer sequences are prone to vanishing gradients, making it harder to learn long term dependencies.

## Linear RNN
Linear RNNs are RNNs without nonlinearity

$$
\begin{aligned}
 h_t &= Ah_{t-1} + Bx_t \\
 y_t &= Ch_t
\end{aligned}
$$

In detail

$$
\begin{aligned}
 h_0&= 0 \\
 h_1 &= Bx_1 \\
 y_1 &= Ch_1 \\
 y_1 &= CBx_1 \\
 y_2 &= CABx_1 + CBx_2 \\
 y_3 &= CA^2Bx_1 + CABx_2 + CBx_3 \\
 y_t &= CA^{t-1}Bx_1 + .......... + CBx_t \\
 K &= (CB, CAB, ...CA^{t-1}B) \\
 y &= K * x
\end{aligned}
$$

Benefits:
* Since convolution is parallelizable, training is fast.



## LSTM
## Transformers
## Mamba
