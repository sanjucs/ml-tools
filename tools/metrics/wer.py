SUB = 1
INS = 1
DEL = 1

def argmin(a):
    return min(range(len(a)), key=lambda x : a[x])

def get_cost(x):
  return x[0] * SUB + x[1] * INS + x[2] * DEL

def set_counts(dp, r, c, prev):
  for i in range(4):
    dp[r][c][i] = prev[i]

def compute_wer(target, predicted, is_case_sensitive=False, debug=False):
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
        set_counts(dp, i, j, dp[i - 1][j - 1])
      else:
        prev_counts = [dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]
        add_costs = [SUB, DEL, INS]
        cost_sub = get_cost(prev_counts[0]) + add_costs[0]
        cost_del = get_cost(prev_counts[1]) + add_costs[1]
        cost_ins = get_cost(prev_counts[2]) + add_costs[2]
        index = argmin([cost_sub, cost_del, cost_ins])

        set_counts(dp, i, j, prev_counts[index])
        dp[i][j][index] += add_costs[index]

  final = dp[-1][-1]
  return sum(final[0:3]) / sum(final)

if __name__ == '__main__':
  target = "The guard arrived late because it was raining"
  predicted = "The guard arrived late because of the rain"

  # target = "A B"
  # predicted = "A D"

  wer = compute_wer(target, predicted)
  print("score::", wer)
