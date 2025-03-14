import torch

class ConfusionMatrix:

  @staticmethod
  def calculate_metrics(confusion_matrix):
    correct = torch.diag(confusion_matrix)
    col_sum = torch.sum(confusion_matrix, dim=0)
    row_sum = torch.sum(confusion_matrix, dim=1)
    precision = correct / col_sum
    recall = correct / row_sum
    f1score = 2 * precision * recall / (precision + recall)
    accuracy = torch.sum(correct) / torch.sum(confusion_matrix)
    balanced_accuracy = torch.mean(recall)
    result = {'confusion_matrix': confusion_matrix, 'precision': precision, 'recall': recall, 'f1score': f1score, 'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy}
    return result

  def compute(self, target, predicted):
    n_classes = len(torch.unique(target))
    confusion_matrix = torch.zeros(n_classes, n_classes)
    for x, y in zip(target, predicted):
      confusion_matrix[x, y] += 1

    return self.calculate_metrics(confusion_matrix)

if __name__ == '__main__':
  torch.manual_seed(0)

  target = torch.randint(0, 3, (16, ))
  predicted = torch.randint(0, 3, (16, ))

  result = ConfusionMatrix().compute(target, predicted)
  print(result)