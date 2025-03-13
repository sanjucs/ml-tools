**Q:** What is a neural network?

**A:** A neural network is a type of machine learning model inspired by the way the human brain works.

---

**Q:** How does a neural network work?

**A:** A neural network model consists of trainable parameters called weights and biases. In the forward pass, the user feeds input data into the model. The model predicts output using the set of weights and biases and computes loss to evaluate the deviation of predicted output from the target using a cost metric commonly known as the loss. In the backpropagation phase, gradients w.r.t the loss is computed and the weights are biases are tuned to minimize the loss.

---

**Q:** What are activation functions in neural networks?

**A:** [Activation functions](/dl/activations.md)

---

**Q:** What is the vanishing gradient problem in neural networks?

**A:** The vanishing gradient problem refers to the phenomenon of diminishing gradients during backpropagation. The issue is prominent in models with a large number of hidden layers, where computing gradients of the initial layers necessitates numerous multiplications. The sigmoid and tanh functions are affected by the vanishing gradient issue since their derivatives lie in the ranges of [0, 0.25] and [0, 1], respectively. Solutions to avoid the vanishing gradient issue:
* Use other activation functions such as ReLU and its variants.
* Use skip or residual connections.

---

**Q:** What is the exploding gradient problem in neural networks?

**A:** The exploding gradient problem refers to the phenomenon when gradients w.r.t. loss become extremely large during backpropagation. Solutions to avoid the exploding gradient problem:
* Normalisation techniques (e.g. Batch Normalization, Layer Normalization, etc.)
* Gradient clipping

---

**Q:** What are the different normalization techniques in neural networks?

**A:** [Normalization techniques](/notes/dl/normalization.md)

---

**Q:** Why is batch normalization not preferred in LLMs?

---

**Q:** Why are the quantization techniques in neural networks?

**A:** [Quantization](/notes/dl/quantization.md)

---

**Q:** What is LoRA finetuning?

**A:** [LoRA](/notes/dl/lora.md)

**Q:** What is distributed training?

**A:** [LoRA](/notes/dl/distributed.md)