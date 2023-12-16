import math

def normalize(sentence):
  return sentence.lower()

class PrecisionNGram:
  @staticmethod
  def calc_score(predicted_count, target_count):
    score = 0
    for word in predicted_count:
      score += min(target_count.get(word, 0), predicted_count[word])
    score = score / sum(predicted_count.values())
    return score

  def get_precision(self, target, predicted, ngram):
    target = normalize(target).split()
    predicted = normalize(predicted).split()

    def _fill(sentence):
      word2count = {}
      for idx in range(0, len(sentence) - (ngram - 1)):
        words = ' '.join(sentence[idx : idx + ngram])
        word2count[words] = word2count.get(words, 0) + 1
      return word2count

    predicted_count = _fill(predicted)
    target_count = _fill(target)

    return self.calc_score(predicted_count, target_count)

class BLEUScore:
  def __init__(self, N):
    self.N = N

  def compute(self, target, predicted):
    precision = 1
    for n in range(1, self.N+1):
      precision *= PrecisionNGram().get_precision(target, predicted, n)
    precision = precision ** (1 / self.N) 

    predicted_len = len(predicted.split(' '))
    target_len = len(target.split(' '))
    brevity_penalty = min(1, math.exp(1 - (target_len/predicted_len)))

    score = precision * brevity_penalty
    return score
 
if __name__ == '__main__':
  target = "The guard arrived late because it was raining"
  predicted = "The guard arrived late because of the rain"

  # target = "The NASA Opportunity rover is battling a massive dust storm on Mars ."
  # predicted = "The Opportunity rover is combating a big sandstorm on Mars ."
  # predicted = "A NASA rover is fighting a massive storm on Mars ."

  score = BLEUScore(4).compute(target, predicted)
  print("score::", score)
