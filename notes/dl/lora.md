# LoRA - Low Rank Adaptation

Large language models are trained with billions of parameters. Fine-tuning methods help to use these pretrained LLMs for other tasks. There are various methods for fine-tuning a model:

* Fine-tune the entire model.
* Fine-tune only a few layers of the model.
* Introduce adaptive layers in the model.
* LoRA

In LoRA, all the pretrained weights are made non-trainable, and a new pair of low-rank matrices is introduced as trainable weights for selected modules.

![LoRA](/notes/dl/assets/lora.png)

$$W_{new} = W_{old} + A \cdot B$$

where $W_{old}$ = pretrained weight, $A$ and $B$ are LoRA A and B weights respectively.

### Pros:
* Since the trainable modules introduced are of lower rank, computations will be faster.
* LoRA weights only occupy a fraction of the size of the whole model.
* During inference $W_{new}$ can be precomputed using $W_{old}$, $A$, and $B$, which helps keep the inference latency the same as the pretrained model.

## Reference
* LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685
* https://www.youtube.com/watch?v=DhRoTONcyZE