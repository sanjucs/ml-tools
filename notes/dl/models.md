# Evolution of AI models

## FC
## CNN
## RNN
At time $t$ layer receives the current input $x(t)$ and previous hidden state $h(t-1)$ and outputs $h(t)$ and $y(t)$. 
Recurrent neural networks (RNNs) are designed to handle sequential data. RNN processes one token at a time, gathering information from all previous inputs in a memory called the hidden state.

$$h(t) = \Phi(W_{hh} h(t-1) + W_{xh} x(t) + b_h)$$
$$y(t) = \Phi(W_{hy} h(t) + b_y)$$

where

$x(t)$ :- input at time $t$

$h(t)$ :- hidden state at time $t$

$y(t)$ :- output predicited at time $t$

$W$ :- Trainable weights

RNNs retain information about previous inputs in the hidden states, which is critical for processing sequential data. However, there are 2 significant issues with RNN.
* Data must be processed sequentially, which drastically reduces the performance.
* The vanishing gradient issue is caused by an increase in the duration of the sequence.

## LSTM
## Transformers
## Mamba