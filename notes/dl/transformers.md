# Transformers

Transformers are the core foundational components of popular LLMs such as GPT and LLaMa. The Google Brain team introduced the transformer architecture in 2017. The model consists of an encoder-decoder structure. The encoder converts the input sequence to a latent space representation z, and the decoder converts the representation to an output sequence. The transformer architecture is as shown in Figure 1.

![Transformers](/notes/dl/assets/transformers.png)

*Figure 1: Figure shows different Transformer architecture [Image credits]:[Transformer paper](https://arxiv.org/abs/1706.03762)*

Main components
## Embedding
For a language modeling task, the input is a sequence of tokens. Embedding helps to construct D-dimensional representations of these tokens. Words that are related will be represented similarly in the embedding space. Embedding weight is a trainable parameter of shape (vocab_size, embedding_dim).
## Postional encoding

## Scaled Dot Product Attention
Attention is the transformer model's key mechanism for selectively and individually focusing on past tokens. Given query, key, and value matrices, attention is computed as follows:

$$ Attention(Q, K, V) = {softmax({{QK^T} \over {\sqrt d_k}}) V}$$

where

$Q:$ Query - What the model is looking for.

$K:$ Key - What each token offer.

$V$ Value - content carried by each token.

$d_k:$ dimensionlity of $K$

Steps:
* Dot product $QK^T$ results a matrix of shape $(N, T, T)$ that measures similarity of each query token with all key tokens.
* For numerical stability, the resultant term is scaled.
* Softmax converts the result into a probability distribution range, which serves as an attention weight. The $t^{th}$ row of this matrix indicates how much attention token t should pay to other tokens.
* Multiplication with $V$ matrix is equivaluent to creating output vector from weighted sum of $V$.

When $Q = K = V$, it is called self attention, otherwise called cross attention.

### Multi-head attention

## Feed Forward Networks
## Auto regressive property
## Complexity analysis
## Reference



* [Attention is all you need](https://arxiv.org/abs/1706.03762)