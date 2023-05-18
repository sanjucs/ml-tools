import math

class PrecisionNGram:
  @staticmethod
  def normalize(sentence):
    sentence = sentence.lower().split()
    return sentence

  @staticmethod
  def calc_score(predicted_count, target_count):
    score = 0
    for word in predicted_count:
      score += min(target_count.get(word, 0), predicted_count[word])
    score = score / sum(predicted_count.values())
    return score

  def get_precision(self, target, predicted, ngram):
    target = self.normalize(target)
    predicted = self.normalize(predicted)

    predicted_count = {}
    for idx in range(0, len(predicted) - (ngram - 1)):
      words = ' '.join(predicted[idx : idx + ngram])
      predicted_count[words] = predicted_count.get(words, 0) + 1

    target_count = {}
    for idx in range(0, len(target) - (ngram - 1)):
      words = ' '.join(target[idx : idx + ngram])
      target_count[words] = target_count.get(words, 0) + 1

    return self.calc_score(predicted_count, target_count)

class BLEUScore:
  @staticmethod
  def compute(target, predicted):
    precision1 = PrecisionNGram().get_precision(target, predicted, 1)
    precision2 = PrecisionNGram().get_precision(target, predicted, 2)
    precision3 = PrecisionNGram().get_precision(target, predicted, 3)
    precision4 = PrecisionNGram().get_precision(target, predicted, 4)
    precision = (precision1 * precision2 * precision3 * precision4) ** 0.25
    predicted_len = len(predicted.split())
    target_len = len(target.split())
    brevity_penalty = min(1, math.exp(1 - (target_len/predicted_len)))
    score = precision * brevity_penalty
    return score
    
if __name__ == '__main__':
  target = "The guard arrived late because it was raining"
  predicted = "The guard arrived late because of the rain"

  target = "The NASA Opportunity rover is battling a massive dust storm on Mars ."
  predicted = "The Opportunity rover is combating a big sandstorm on Mars ."
  # predicted = "A NASA rover is fighting a massive storm on Mars ."

  score = BLEUScore.compute(target, predicted)
  print("score::", score)
