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

### Parameters affecting performance metrics

#### Scheduling schemes
The scheuler optimize the batching and it's management with regard to computation and memory constraints. In LLM, there are various scheduling schemes.
<noformat>
* Static Request-level scheduling\
	Batching is done in the order that the requests arrive. When the batch is ready, it begins model execution. If a new request comes in the middle of current execution, it must wait until the existing one is completed.
* Iteration level\
	Iteration level is considered as the level at which single token is generated. If one of the requests in the current batch has finished decode phase, new request's prefill phase can be added to the existing batch. However, if the iteration includes both prefill and decode queries, the decode requests must be heavily padded.
* Packaged batching\
	Instead of batching dimension, packaged batching appends the requests to the sequence dimention, addressing padding issue.
* Continuous batching\
	Continuous batching (in-fligh batching) is a combination of iteration level and packaged batching.
* Memory aware scheduling\
	Pre-allocation and On-demand scheduling -	KV caches play a key role in scheduling. If the device memory required for KV cache exceeds the limit, preemptive measures such as recomputation or swap will take place for the least used requests, resulting in a performance decrease. Pre-allocation solves this problem by allocating memory for requests during the initialization phase, but resulting smaller running batch size. As the name implies, on-demand scheduling allocates KV caches during runtime.
<noformat>

Note: All requests should be padded to match the length of longest sequence in a single batch.

#### Batch size and max num of tokens
Increasing batch size allows more tokens to be processed simultaneously. This improves compute utilizations in prefill phase and minimizes TTFT, increasing throughput. In contrast, increasing batch size causes the decode phase's iteration time to slow down, resulting in a higher TPOT. Prefill batch size will be limied by max number of tokens.

## Reference
* https://blog.squeezebits.com/
