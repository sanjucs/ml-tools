# Normalization techniques

![Normalization](/notes/dl/assets/normalization.jpg)
*Figure 1: Figure shows different normalization techniques on a tensor with shape (N, C, H, W). [Image credits]:[Group normalization](https://arxiv.org/pdf/1803.08494)*
## Batch normalization

Batch normalization is a technique for normalizing the activation, ensuring the mean and variance are 0 and 1, respectively. This helps to mitigate the vanishing and exploding gradient problems.

### Steps during training

* For each mini-batch, calculate the mean and variance for each feature.

	$\mu = {1 \over N} {\sum_{i}x_i}$

	$σ^2 = {1 \over N } {\sum_{i}(x_i-\mu)^2}$ where $N$ is the total number of datapoints.


* Normalize the value by subtracting the mean from the value and then dividing it by the square root of variance.

	$\hat{x_i} = {x_i -\mu \over \sqrt{σ^2 + \epsilon}}$ where $\epsilon$ is as small positive number added to prevent numerical instability.

* Apply the learning parameters gamma and beta on the normalized value.

	$y_i = \gamma \cdot \hat{x_i} + \beta$

* Compute the weighted running mean and variance during training to be stored and used during inference.

	$\mu_{new} = \mu_{curr} \cdot \alpha + \mu_{old} \cdot (1 - \alpha)$ where

	$\mu_{new}$ = updated value

	$\mu_{curr}$ = new observed value

	$\mu_{old}$ = previous estimated value

	$\alpha$ = factor for the weighted sum

	σ is also calculated in a similar fashion.

### Steps during inference
* Normalize the value using the running mean and running variance computed during training.
* Apply the learning parameters gamma and beta on the normalized value.

Batch normalization generally used for CNN and FC models and not preferred in LLMs.

## Layer normalization
The layer norm calculates the normalization statistics for each data sample in the layer separately. The layer normalization method doesn't rely on mini-batch size and doesn't necessitate the calculations of running statistics for inference.

## Instance normalization
The instance norm is a normalization process that runs for each feature and each data sample in the mini-batch.

## Group normalization
The group normalization technique is defined as a normalization technique that falls somewhere between layer and instance normalizations. Instead of calculating normalization statistics for each feature, the group norm calculates them for a group of features.

---
## Reference
* Group normalization (Paper Explained) https://www.youtube.com/watch?v=l_3zj6HeWUE
