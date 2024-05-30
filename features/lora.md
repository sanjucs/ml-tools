# LoRA - Low Rank Adaptation

Language models are trained with billions of parameters. To utilize the models for own stream tasks, we need to fine-tuning methods. A few of them listed below.
* Fine-tune the entire model
* Fine-tune only a few layers of the model
* Introduce adaptive layers in the model.
* LoRA

## LoRA
In LorA, all pretrained weights are made non-trainable and introduces a new pair of trainable weights for selective modules with lower rank.

![Figure 1](/features/assets/lora.png) 

$W_{new} = W_{old} + AB$

where 
$W_{old}$ = pretrained weight, $A$ and $B$ are LoRA A and B weights respectively.

### Pros:
* Since the trainable modules introduced are with lower rank, computation will faster.
* Require less storage since only need to save new LoRA modules and pretrained modules remain intact.
* During inference $W_{new}$ can be precomputed using $W_{old}$, $A$ and $B$ which avoids inference latency.

## Reference

* LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685
* https://www.youtube.com/watch?v=DhRoTONcyZE 
