import torch
import math

def compute_softmax(x, is_partial=False):
  x_max = torch.max(x)
  x_exp = torch.exp(x - x_max)
  x_exp_sum = torch.sum(x_exp)

  if is_partial:
    return x_exp / x_exp_sum, x_max, x_exp_sum
  else:
    return x_exp / x_exp_sum
 
def softmax(x, block_size=None):
  if block_size == None or block_size > len(x):
    return compute_softmax(x)

  n_splits = math.ceil(len(x) / block_size)
  running_m = running_l = 0
  out = torch.zeros(len(x))
  for i in range(n_splits):
    start = block_size * i
    end = min(len(x), block_size * (i + 1))
    out_curr, m_curr, l_curr = compute_softmax(x[start : end], is_partial=True)

    m_max = max(m_curr, running_m)

    l_factor1 = running_l * torch.exp(running_m - m_max)
    out = out * l_factor1
    l_factor2 = l_curr * torch.exp(m_curr - m_max)
    out[start:end] = out_curr * l_factor2

    running_l = l_factor1 + l_factor2
    running_m = m_max
    out = out / running_l

  return out

if __name__ == '__main__':
  a = torch.rand((50))

  out = torch.softmax(a, dim=0)
  out1 = compute_softmax(a)
  out2 = softmax(a, 3)

  print(torch.allclose(out, out1))
  print(torch.allclose(out, out2))
