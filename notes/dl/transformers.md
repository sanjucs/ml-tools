# Transformers

Transformers are the core foundational components of popular LLMs such as GPT and LLaMa. The Google Brain team introduced the transformer architecture in 2017. The model consists of an encoder-decoder structure. The encoder converts the input sequence in a latent space representation z, and the decoder converts the representation to an output sequence. The transformer architecture is as shown in Figure 1.

![Transformers](/notes/dl/assets/transformers.png)

*Figure 1: Figure shows Transformer architecture [Image credits]:[Transformer paper](https://arxiv.org/abs/1706.03762)*

Main components
## Embedding
For a language modeling task, the input is a sequence of tokens. The embedding layer converts integer tokens to real valued vectors. Embedding weight is a trainable parameter of shape (vocab_size, embedding_dim).

## Postional encoding

## Scaled Dot Product Attention
Attention is the transformer model's key mechanism for selectively and individually focusing on past tokens. Given query, key, and value matrices, attention is computed as follows:

$$ Attention(Q, K, V) = {softmax({{QK^T} \over {\sqrt d_k}}) V}$$

where

$Q:$ Query - What the model is looking for.

$K:$ Key - What each timestep offer.

$V$ Value - content carried by each timestep.

$d_k:$ dimensionlity of $K$

Steps:
* Dot product $QK^T$ results in a matrix of shape $(N, T, T)$ that measures the similarity of each query token with all key tokens.
* For numerical stability, the resultant term is scaled.
* Softmax converts the result into a probability distribution range, which serves as an attention weight. The $t^{th}$ row of this matrix indicates how much the attention token t should pay to other tokens.
* Multiplication with the $V$ matrix is equivaluent to creating output vector from weighted sum of $V$.

When $Q = K = V$, it is called self attention, otherwise called cross attention.

### Multi-head attention
Multi-head attention executes numerous attention functions with $d_{model} / h$ dimension as opposed to a single attention function with $d_{model}$ dimensions. Because each head operates independently, multi-head attention can be computed in parallel.

$$ MultiHead(Q, K, V) = Concat(head_1, head_2, head_3, head_h)$$

where

$$head_i = Attention(Q_iW^Q_i, K_iW^K_i, V_iW^V_i)$$

## Feed Forward Networks

FFN introduces non-linearity in the transformer architecture and helps to learn more complex patterns.

$$ FFN(x) = \phi(0,xW_1 + b)W_2 + b $$

## Auto regressive property

To preserve auto-regressive property during training, leftward information flow is prevented by masking out those values prior to applying softmax.

## Complexity analysis

* Compute complexity correlates to the number of MAC operations required to pcocess all T tokens.
* Time complexity estimates how long the system takes to compute the outputs. This can be measured in two ways
    * Non-parallelized (NP): The time it takes if the all the computations are done using a single process.
    * Parallelized (P): The time it takes if all independent computations are allowed to run on parallel processes. This is the key to understanding why transformers are faster, because GPUs are designed to run things parallelly.
* Space complexity is indicative of the RAM requirements.

$T:-$ Sequence length

$d:-$ hidden dim

| Component             | Time Complexity (NP)   | Time Complexity (P)   | Compute Complexity         | Space Complexity         |
| --------------------- |------------------------| -------------------| --------------------------| --------------------------| 
| Attention             | $\mathcal{O}(T^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(T^2 \cdot d)$   | $\mathcal{O}(T^2)$          |
| Feedforward Layer     | $\mathcal{O}(T)$ | $\mathcal{O}(1)$ | $\mathcal{O}(T \cdot d^2)$   | $\mathcal{O}(T \cdot d)$     |
| Transformer Layer     | $\mathcal{O}(T^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(T^2 \cdot d + T \cdot d^2)$ | $\mathcal{O}(T^2 + T \cdot d)$ |
| RNN Layer             | $\mathcal{O}(T)$ | $\mathcal{O}(T)$ | $\mathcal{O}(T \cdot d)$ | $\mathcal{O}(T \cdot d)$ |

Here "transformer layer" is just a combination ofr the "attention" and "feedforward layer". From the time-complexity analysis with the non-parallel approach, even though it looks like transformers require more compute, the key here is that all those processes are deliberately designed to be parallelized to make the compute highly efficient on GPUs. Therefore, transformers have an effective contant time-complexity, which basically means everything in a layer can be computed in the same speed regardless of the length of the sequence (Note that this is true only if there is infinite compute. In reality, we will still be bottlenecked by our hardware limits of how many processes we can run in parallel). Unlike transformers, RNNs were not designed with GPUs in mind. Therefore, even if there are plenty of cores to process parallel requests, it is impossible to process then parallelly in RNNs, since the next timestep can only be processes after the current timestep's computations have finished. This makes RNNs inefficient on GPUs.

### Training phase
For a sequence-to-sequence task during training, we have the input sequence and the target sequence ready. The whole of the input sequence can be passed through the encoder and be run parallely. The decoder however is auto-regressive, which basically means that the next timestep is conditioned on the previous prediction. This gives the impression that we need to wait for the processing to be done sequentially in the decoder. But the trick here is that during training, we know what we desire each timestep to predict. Instead of looping in the prediction from the previous timestep of the decoder, we can provide the true targets we desire from the previous timestep and remove the sequential requirements. This makes the processing in the decoders $\mathcal{O}(T^2)$ during training.

### Inference phase
Transformers shine on GPUs when we have everything ready to be processed. During inference, this is usually not the case since the ouput tokens are not ready and we generate them one-by-one (Next Token Prediction). Take the example of a machine translation task. We have the tokens of the source language eady to be processed. The encoder can be processed parallely, since all the information required to do parallel processing is ready. When it comes to the decoder, unlike during training, we only have the first token which is usually the $<SOS>$ token. Every new token has to be processed sequentially after the previous token has been generated. If done naively by processing everything in the decoder again, this would mean that for every new token we should do an $\mathcal{O}(T^2)$ computation. But if we cache the key and value vectors computed for the previous timesteps (which don't change since the decoder has causal attention), for every new token we only need to compute the $q$, $k$ and $v$ vectors for that particular timeframe and then do the attention operation. This technique is commonly called KV-caching. Even without parallel processing, for every new token this is only $\mathcal{O}(T)$, and with parallel processing it becomes $\mathcal{O}(1)$. Therefore for an average output sequence of length $T$, the time-complexity after parallelizations is $\mathcal{O}(T)$. Note that this is not better than RNNs in terms of speed, since RNNs can also do processing at the same speed during auto-regression. Unlike RNNs which only needs to hold a single vector for the previous hidden step, transformers need to hold all the previous $k$ and $v$ vectors which increases as the length of the decoding sequence increases. Therefore in terms of just time-complexity, transformers and RNNs are equivalent during a NAR decode with $\mathcal{O}(T)$ and in terms of space-complexity, RNNs require only $\mathcal{O}(d)$ while transformers need $\mathcal{O}(T \cdot d)$.

## Reference
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
