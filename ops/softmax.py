import torch
import math

def compute_softmax(x, dim, is_partial=False):
  x_max = torch.max(x, dim=dim, keepdim=True)[0]
  x_exp = torch.exp(x - x_max)
  x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)

  if is_partial:
    return x_exp / x_exp_sum, x_max, x_exp_sum
  else:
    return x_exp / x_exp_sum
 
def softmax(x, dim, block_size=None):
  if block_size == None or block_size > x.shape[dim]:
    return compute_softmax(x, dim)

  n_splits = math.ceil(x.shape[dim] / block_size)
  running_sum = running_max = None
  running_out = torch.zeros(x.shape)

  for i in range(n_splits):
    start = block_size * i
    end = min(x.shape[dim], block_size * (i + 1))
    x_slice = x.transpose(0, dim)[start:end].transpose(0, dim)
    block_out, block_max, block_sum = compute_softmax(x_slice, dim=dim, is_partial=True)

    if running_max is None or running_sum is None:
      running_max = torch.zeros_like(block_max)
      running_sum = torch.zeros_like(block_max)

    global_max = torch.max(torch.stack([block_max, running_max]), dim=0)[0]
    sum_prev = running_sum * torch.exp(running_max - global_max)
    sum_curr = block_sum * torch.exp(block_max - global_max)

    running_out = running_out * sum_prev
    running_out = running_out.transpose(0, dim)
    running_out[start:end, ...] = (block_out * sum_curr).transpose(0, dim)
    running_out = running_out.transpose(0, dim)

    running_sum = sum_prev + sum_curr
    running_max = global_max
    running_out = running_out / running_sum
  return running_out

if __name__ == '__main__':
  a = torch.rand((200, 100, 20))

  dim = 1
  running_out = torch.softmax(a, dim=dim)
  out1 = compute_softmax(a, dim)
  print(torch.allclose(running_out, out1))

  out2 = softmax(a, dim=dim, block_size=9)
  print(torch.allclose(running_out, out2))
