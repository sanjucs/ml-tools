# LLM

## Phases of text generation
* Prefill phase
	* Processes all input tokens, creates contexts and generate first output token.
	* Computationally intensive since the all the inputs are fed to the model at a time.
* Decode phase
	* Generates next output token auto-regressively.
	* Utilizes the KV-cache created during prompt phase to avoid redundant compuations for subsequent token generations. However this introduces additional memory transfer costs.

## Sampling methods
Objective: Generate tokens with diversiry and coherence.
* Greedy apporach - Selecting token with highest probability. Faces degeneration issue (repetitive).
* Top K - Randomly choose token from K number of with highest probabilites.
* Top P - Randomly choose token from tokens with cumulative probability greater than p.
* Repetition penalty


## Performance
The performance of the LLMs can be evaluated using the following metrics.

* Throughput: Total number of tokens / Total ineference time
* TTFT (Time Taken for First Token): Latency measured between the request and first token generated.
* TPOT (Time Per Output Token)

Parameters affecting performance metrics:
* Batch size
	* Increasing batch size allows more tokens to be processed simultaneously. This results in more compute utilization and reduces TTFT, thus increasing throughput. However, the upper limit of batch size is limited by memory constraints. As batch size increases, kv cache allocations are increased, which might exceed the memory limit and eventually result in preempt techniques like recomputation or swap.
	* Increasing match size slows-down iteration time in decode phase which results increase in TPOT.
* Max num of tokens tokens
* Request rate

Reference
1. https://blog.squeezebits.com/
