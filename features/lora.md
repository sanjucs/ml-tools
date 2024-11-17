# LoRA - Low Rank Adaptation

Large language models are trained with billions of parameters. To utilize the models for other tasks, we need fine-tuning methods. There are various methos for fine-tuning a model.
* Fine-tune the entire model
* Fine-tune only a few layers of the model
* Introduce adaptive layers in the model.
* LoRA

## LoRA
In LorA, all pretrained weights are made non-trainable and a new pair of low-rank matrices are introduced as trainable weights for selected modules.

![Figure 1](/features/assets/lora.png) 

$W_{new} = W_{old} + AB$

where 
$W_{old}$ = pretrained weight, $A$ and $B$ are LoRA A and B weights respectively.

### Pros:
* Since the trainable modules introduced are of lower rank, computations will be faster.
* Requires less storage since we only need to save the new LoRA wrights, while the pretrained weights remain intact.
* During inference $W_{new}$ can be precomputed using $W_{old}$, $A$ and $B$ which helps keeps the inference latency the same as pretrained model.

## Reference

* LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685
* https://www.youtube.com/watch?v=DhRoTONcyZE