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

The confusion matrix is used in multi-class classification tasks to evaluate a system's performance. It is an $(N, N)$ matrix, with each row representing the target class and each column representing the predicted class. The diagonal elements of the matrix correspond to the correctly predicted data samples. The metrics mentioned in the previous section can be easily calculated using the confusion matrix.

Let A be the confusion matrix; then precision, recall and accuracy of the $i^{th}$ class can be calculated as

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
The implementation for the confusion matrix can be found [here](/tools/metrics/confusion_matrix.py).

## Regression tasks
$$ \text{Mean Square Error = } MSE = {1 \over N} \sum_i(y_i - \hat{y_i})^2$$

$$ \text{Root mean Square Error = } RMSE = \sqrt {{1 \over N} \sum_i(y_i - \hat{y_i})^2}$$

$$ \text{Mean Absolute Error = } MAE = {1 \over N} \sum_i|
y_i - \hat{y_i}|$$

where

$N$: number of samples

$\hat{y_i}$: actual value

$y_i$: predicted value

## Performance metrics for NLP tasks

<ins> BLEU Score </ins>

For a text translation scenario, the range of acceptable answers is large. This makes evaluation of the predicted result challenging. BLEU score is an abbreviation for Bilingual evaluation understudy, and it was developed by IBM in 2001 to assess the quality  machine translated text. It returns a value in range of $[0, 1]$ indicating how similar the predicted text is to the target text. BLEU score is defined as

$$BLEU(N) = \text{Brevity Penalty} \cdot \text{Geometric Average precision Score(N)}$$

$\text{Brevity Penalty} = min(1, e^{1 - {\text{target len} \over \text{predicted len}}})$

$\text{Geometric Average precision Score(N)} = \prod_{n=1}^{N} P_{n}^{w_{n}}$

where

$N$: $n-gram$

$P_{n}$: Precision of $n-gram$

$w_{n}$: weightage of $n-gram$

The brevity penalty helps to penalize the sentences that are too short. The implementation for BLEU Score can be found [here](/tools/metrics/bleu_score.py).

Example:
```
target =    "The guard arrived late because it was raining"
predicted = "The guard arrived late because of the rain"

Calculate BLEU(4)?

target length = 8
predicted length = 8
Brevity penalty = 1

Precision-1 gram = 5/8 = 0.625
Precision-2 gram = 4/7 = 0.571
Precision-3 gram = 3/6 = 0.5
Precision-4 gram = 2/5 = 0.4
Geometric Average precision Score(N) = 0.517

BLEU(N) = 0.517
```
Cons:

* Does not consider the meaning or order of the word.
* Searches for exact word matches. Fails to explore synonyms or alternate forms.
* Neglects to emphasize essential words.

<ins> Word Error Rate (WER) </ins>

WER is a metric used to evaluate the predictions in ASR and machine translational systems. The predicted text may contain new words, missing words, or alternate words. WER takes into account all of these aspects and calculates the error accordingly.

$$WER = {S + I + D \over N}$$

where

$S$: number of substitutions

$I$: number of insertions

$D$: number of deletions

$N$: number of words in target

Note: WER tries to align target to predicted text. 

The implementation for the wer can be found [here](/notes/dl/modules/metrics/wer.py).

Example:

![WER](/notes/dl/assets/wer.png)

## Reference
* [[Wiki] Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
* [BLEU score](https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b/)