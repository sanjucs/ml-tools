def compute_wer(target, predicted):
  target = target.lower().split()
  predicted = predicted.lower().split()

  predicted_len = len(predicted)
  target_len = len(target)

  dp = [[0] * (predicted_len + 1) for _ in range(target_len + 1)]

  for i in range(1, predicted_len + 1):
    dp[0][i] = i

  for i in range(1, target_len + 1):
    dp[i][0] = i

  for i in range(1, target_len + 1):
    for j in range(1, predicted_len + 1):
      if target[i-1] == predicted[j-1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = min(min(dp[i][j-1], dp[i-1][j]), dp[i-1][j-1]) + 1

  return dp[-1][-1] / target_len

if __name__ == '__main__':
  target = "The guard arrived late because it was raining"
  predicted = "The guard arrived late because of the rain"

  wer = compute_wer(target, predicted)
  print("score::", wer)
