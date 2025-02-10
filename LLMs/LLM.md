# LLM

## Phases of text generation
There are mainly two phases in text generation: the prefill phase and the decode phase. The prefill phase processes all the input tokens creates contexts, and generates the first output token. This phase is computationally intensive since all the inputs are fed to the model simultaneously. The decode phase generates the next output token auto-regressively. It utilizes the KV cache created during the prompt phase to avoid redundant computations for subsequent token generations. However, this introduces additional memory transfer costs.

## Sampling methods
The objective of the sampling technique is to generate tokens with diversity and coherence. Below are a few of the sampling techniques.
* Greedy approach - Selecting token with the highest probability. Often faces degeneration issues.
* Top K - Randomly choose a token from the K number of tokens with the highest probabilities.
* Top P - Randomly choose the token from tokens with cumulative probability greater than p.
* Repetition penalty

## Performance
The performance of the LLMs can be evaluated using the following metrics.
* Throughput: Total number of tokens / Total inference time
* TTFT (Time Taken for First Token): Latency measured between the request and first token generated.
* TPOT (Time Per Output Token)

Parameters affecting performance metrics:
* Batch size
	* Increasing batch size allows more tokens to be processed simultaneously. This results in more compute utilization and reduces TTFT, thus increasing throughput. However, the upper limit of batch size is limited by memory constraints. As batch size increases, kv cache allocations are increased, which might exceed the memory limit and eventually result in preempt techniques like recomputation or swap.
	* Increasing match size slows-down iteration time in decode phase which results increase in TPOT.
* Max num of tokens tokens
* Request rate

## Reference
* https://blog.squeezebits.com/
