# LLM

## Phases of text generation
* Prefill phase
	* Processes all input tokens, creates contexts and generate first output token.
	* Computationally intensive since the all the inputs are fed to the model at a time.
* Decode phase
	* Generates next output token auto-regressively.
	* Utilizes the KV-cache created during prompt phase to avoid redundant compuations in the subsequent token generations. However this introduces additional memory transfer costs.


## Performance
The performance of the LLMs can be evaluated using the following metrics.

* Throughput: Total number of tokens / Total ineference time
* TTFT (Time Taken for First Token): Latency measures between the request and first token generated.
* TPOT (Time Per Output Token)

Parameters affecting performance metrics:
* Batch size
	* Increasing batch size causes more compute utilization => increases throughput.
	* Lower batch size causes faster iteration in decode phase => decreases TPOT.
* Request rate

Reference
1. https://blog.squeezebits.com/