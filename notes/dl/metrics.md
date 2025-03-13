# Accuracy metrics

In a neural network, accuracy metrics are used to evaluate the correctness of the model trained. The metrics are being selected based on the type of problem and data.

## Classification tasks
In the classification tasks, accuracy metrics are evaluated based on the target class and predicted class of each sample. A few of the commonly used metrics are listed below.

$$ \text{accuracy} = {\text{total number of correctly predicted samples} \over \text{total number of samples}}$$

$$ \text{precision of class } C = {\text{number of correctly predicted samples in class }C \over \text{number of samples predicted class }C}$$

$$ \text{recall of class } C = {\text{number of correctly predicted samples in class }C \over \text{number of samples in class }C}$$

The phrases used to denote the correctness of each data sample are as follows:

$TP$: True Positive (correctly predicted positive samples).

$TN$: True Negative (correctly predicted negative samples).

$FP$: False Positive (Incorrectly predicted negative samples / false alarms).

$FN$: False Negative (correctly predicted positive samples - misses/underestimations).

For a 2 class problem
$$ accuracy = {TP + TN \over TP + TN + FP + FN}$$

$$ precision = { TP \over TP + FP}$$

$$ recall = { TP \over TP + FN}$$

## Reference
* [[Wiki] Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
