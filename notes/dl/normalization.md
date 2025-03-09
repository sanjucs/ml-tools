# Normalization techniques

## Batch normalization

Batch normalization is a technique for normalizing the output of the previous layer, ensuring the mean and variance are 0 and 1, respectively. This helps to speed up the training and mitigates the vanishing and exploding gradient problems.

### Steps during training

* For each mini-batch, calculate the mean and variance for each feature.

	$\mu = \sum_{i}x_i$

	$σ^2 = {\sum_{i}(x_i-\mu)^2 \over N}$ where $N$ is the total number of datapoints.


* Normalize the value by subtracting the mean from the value and then dividing it by the square root of variance.

	$\hat{x_i} = {x_i -\mu \over \sqrt(σ^2 + \epsilon)}$ where $\epsilon$ is as small positive number added to prevent numerical instability.

* Apply the learning parameters gamma and beta on the normalized value.

	$y_i = \gamma \cdot \hat{x_i} + \beta$

* Compute the running mean and running variance, which are used during inference.

	$x_{new} = x_{curr} \cdot \alpha + x_{old} \cdot (1 - \alpha)$ where

	$x_{new}$ = updated value

	$x_{curr}$ = new observed value

	$x_{old}$ = previous estimated value

	$\alpha$ = factor for the weighted sum

### Steps during inference
* Normalize the value using the running mean and running variance computed during training.
* Apply the learning parameters gamma and beta on the normalized value.

Batch normalization generally used for CNN and FC models and not preferred in LLMs.
## Reference
* Group Normalization (Paper Explained) https://www.youtube.com/watch?v=l_3zj6HeWUE
