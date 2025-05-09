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
Multi-head attention executes numerous attention functions with $d_{model} / h$ dimension as opposed to a single attention function with $d_{model}$ dimension. Because each head operates independently, multi-head attention can be computed in parallel, making it faster.

$$ MultiHead(Q, K, V) = Concat(head_1, head_2, head_3, head_h)$$

where

$$head_i = Attention(Q_iW^Q_i, K_iW^K_i, V_iW^V_i)$$

## Feed Forward Networks
## Auto regressive property
## Complexity analysis

* Compute complexity estimates how much MAC operations required to pcocess all T tokens.
* Time complexity estimates how compute complexity scales as more tokens as added.
* Memory complexity tracks memory reuirements in RAM.

$T:-$ Sequence length

$d:-$ hidden dim

| Component             | Time Complexity        | Compute Complexity         | Memory Complexity         |
| --------------------- |------------------------| -------------------| --------------------------| 
| Attention             | $\mathcal{O}(T^2)$ | $\mathcal{O}(T^2 \cdot d)$   | $\mathcal{O}(T^2)$          |
| Feedforward Layer     | $\mathcal{O}(T)$ | $\mathcal{O}(T \cdot d^2)$   | $\mathcal{O}(T \cdot d)$     |
| Transformer Layer     | $\mathcal{O}(T^2)$ | $\mathcal{O}(T^2 \cdot d + T \cdot d^2)$ | $\mathcal{O}(T^2 + T \cdot d)$ |

Attention weights can be computed in parallel (with equivalent of constant time complexity) because all tokens are already available during training. However, because inference is auto-regressive, attention weights of previously generated tokens must be recalculated everytime in order to construct a new token. With a time complexity of $\mathcal{O}(T)$, this limits the speed. This problem can be fixed by caching the created tokens' key and value vectors (KV caching), however doing so comes at a high memory cost.
## Reference



* [Attention is all you need](https://arxiv.org/abs/1706.03762)