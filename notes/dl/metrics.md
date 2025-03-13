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

<ins>Confusion matrix</ins>

The confusion matrix is a matrix to evaluate the performance of a model by comparing the target and predicted values. It is an $(N, N)$ matrix, with each row representing the target class and each column representing the predicted class. The diagonal elements of the metrics correspond to the correctly predicted data samples. The metrics mentioned in the previous section can be easily calculated using the confusion matrix.

Let A be the confusion matrix; then precision, recall and accuracy of the $i^{th}$ class are calculated as

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
f1-score  : [0.6667, 0.2857, 0.3077]
accuracy  : 0.4375
balanced_accuracy': 0.4222
```
The implementation for the confusion matrix can be found [here](/notes/dl/modules/metrics/confusion_matrix.py).

## Regression tasks
$$ \text{Mean Square Error = } MSE = {1 \over N} \sum_i(y_i - \hat{y_i})^2$$

$$ \text{Root mean Square Error = } RMSE = \sqrt {{1 \over N} \sum_i(y_i - \hat{y_i})^2}$$

$$ \text{Mean Absolute Error = } MAE = {1 \over N} \sum_i|
y_i - \hat{y_i}|$$

where

$N$: number of samples

$\hat{y_i}$: actual value

$y_i$: predicted value
## Reference
* [[Wiki] Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
