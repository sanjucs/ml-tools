# Performance metrics

In a neural network, performance metrics are used to evaluate the correctness of the model trained. The metrics are being selected based on the type of problem and data.

## Classification tasks
In the classification tasks, the performance is evaluated based on the target class and predicted class of each sample. A few of the commonly used metrics are listed below.

$$ \text{accuracy} = {\text{total number of correctly predicted samples} \over \text{total number of samples}}$$

$$ \text{precision of class C} = {\text{number of correctly predicted samples in class C} \over \text{number of samples predicted class C}}$$

$$ \text{recall of class C} = {\text{number of correctly predicted samples in class C} \over \text{number of samples in class C}}$$

$$\text{F1-score} = {2 \over {1 \over precision} + {1 \over recall}}$$
$$\text{F1-score} = {2 \cdot precision \cdot recall \over precision + recall}$$

The phrases used to denote the correctness of each data sample are as follows:

$TP$: True Positive (correctly predicted positive samples).

$TN$: True Negative (correctly predicted negative samples).

$FP$: False Positive (Incorrectly predicted negative samples / false alarms).

$FN$: False Negative (correctly predicted positive samples - misses/underestimations).

For a 2 class problem:

$$ accuracy = {TP + TN \over TP + TN + FP + FN}$$

$$ precision = { TP \over TP + FP}$$

$$ recall = { TP \over TP + FN}$$

<u>Confusion matrix</u>

The confusion matrix is a matrix to evaluate performance of model by comparing target value to predicted value. It is a $(N, N)$ matrix in which each row represent the target class and each colomn represents the predicted class. The diagonal elemnts of the metrics represents the correctly predicted data samples. The metrics defined in the previous section can be easily computed from confusion matrix.

Let A be the confusion matrix, then precision and recall of the $i^{th}$ class is defined as

$$precision_{i} = {A_{ii} \over \sum_{j} Aji}$$

$$recall_{i} = {A_{ii} \over \sum_{j} Aij}$$

$$accuracy = {\sum_{i} Aii \over \sum_{i}\sum_{j} Aij}$$

Example:
```
target    = [2, 0, 2, 0, 1, 0, 1, 1, 1, 0, 2, 2, 0, 0, 1, 2]
predicted = [0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 1, 2, 0, 0, 1, 0]

confusion matrix = [[4., 0., 2.],
					[0., 1., 4.],
					[2., 1., 2.]] (target x predicted)

precision : [0.6667, 0.5000, 0.2500] 
recall    : [0.6667, 0.2000, 0.4000] 
f1-score   : [0.6667, 0.2857, 0.3077]
accuracy  : 0.4375
balanced_accuracy': 0.4222
```
The implementation for the confusion matrix can be found [here](/notes/dl/modules/metrics/confusion_matrix.py).

## Reference
* [[Wiki] Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
