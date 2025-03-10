# INT8 quantization - Linear quantization
import torch

def linear_quantize(tensor, dtype=torch.int8):
  r_min, r_max = tensor.min().item(), tensor.max().item()
  q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

  scale = (r_max - r_min) / (q_max - q_min)
  zero_point = q_min - (r_min / (scale + 1e-9))

  if zero_point < q_min:
    zero_point = q_min
  elif zero_point > q_max:
    zero_point = q_max

  zero_point = int(round(zero_point))

  quantized_tensor = tensor / scale + zero_point
  quantized_tensor = quantized_tensor.clamp_(q_min, q_max)

  return quantized_tensor, scale, zero_point

def linear_dequantize(qtensor, scale, zero_point):
  tensor = scale * (qtensor - zero_point)
  return tensor

if __name__ == '__main__':
  input_tensor = torch.randn(4, 4)
  q_tensor, scale, zero_point = linear_quantize(input_tensor)
  dq_tensor = linear_dequantize(q_tensor, scale, zero_point)
  q_error = (input_tensor - dq_tensor).square().mean()

  print(q_error)