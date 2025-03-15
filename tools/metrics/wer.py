class WER:
  def __init__(self, SUB=1, INC=1, DEL=1):
    self.SUB = SUB
    self.INC = INC
    self.DEL = DEL

  @staticmethod
  def argmin(a):
      return min(range(len(a)), key=lambda x : a[x])

  def get_cost(self, x):
    return x[0] * self.SUB + x[1] * self.INC + x[2] * self.DEL

  @staticmethod
  def set_counts(dp, r, c, prev):
    for i in range(4):
      dp[r][c][i] = prev[i]

  def _compute(self, target, predicted, is_case_sensitive=False):
    if not is_case_sensitive:
      target = target.lower()
      predicted = predicted.lower()

    target = target.split()
    predicted = predicted.split()

    predicted_len = len(predicted)
    target_len = len(target)

    dp = [[[0] * 4 for _ in range(predicted_len + 1)] for _ in range(target_len + 1)]

    for i in range(1, predicted_len + 1):
      dp[0][i][1] = i

    for i in range(1, target_len + 1):
      dp[i][0][2] = i

    for i in range(1, target_len + 1):
      for j in range(1, predicted_len + 1):
        if target[i-1] == predicted[j - 1]:
          self.set_counts(dp, i, j, dp[i - 1][j - 1])
          dp[i][j][3] += 1
        else:
          prev_counts = [dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]
          add_costs = [self.SUB, self.DEL, self.INC]
          cost_sub = self.get_cost(prev_counts[0]) + add_costs[0]
          cost_del = self.get_cost(prev_counts[1]) + add_costs[1]
          cost_INC = self.get_cost(prev_counts[2]) + add_costs[2]
          index = self.argmin([cost_sub, cost_del, cost_INC])

          self.set_counts(dp, i, j, prev_counts[index])
          dp[i][j][index] += add_costs[index]

    return dp[-1][-1]

  def cost(self, target, predicted, is_case_sensitive=False):
    x = self._compute(target, predicted, is_case_sensitive)
    return sum(x[:-1]) / sum(x)
    
  def SDIC(self, target, predicted, is_case_sensitive=False):
    x = self._compute(target, predicted, is_case_sensitive)
    return {'S': x[0], 
            'D': x[1],
            'I': x[2],
            'C': x[3],
            }

if __name__ == '__main__':
  target = "The guard arrived late because it was raining"
  predicted = "The guard arrived late because of the rain"

  wer = WER().cost(target, predicted)
  print("score::", wer)

  SDIC = WER().SDIC(target, predicted)
  print("SDIC::", SDIC)
