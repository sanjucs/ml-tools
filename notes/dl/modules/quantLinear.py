import torch

from quantization import linear_sym_quantize, linear_dequantize

class QLinear(torch.nn.Module):
  def __init__(self, in_features, out_features, bias=False, dtype=torch.float32):
    super(QLinear, self).__init__()

    self.register_buffer("int8_weights", torch.randint(-128, 128, (out_features, in_features), dtype=torch.int8))

    self.register_buffer("int8_scales", torch.randn(1, dtype=dtype))

    if bias:
      self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
    else:
      self.bias = None

  def quantize(self, weights):
    self.int8_weights, scale = linear_sym_quantize(weights)
    self.int8_scales = torch.tensor(scale)

  def dequantize(self):
    return linear_dequantize(self.int8_weights, self.int8_scales)

  def forward(self, x):
    weights = self.dequantize()
    weights = weights.to(x.dtype)

    x = torch.nn.functional.linear(x, weights)

    if self.bias is not None:
      x = x + self.bias

    return x

if __name__ == '__main__':
  in_features = 12
  out_features = 5
  batch_size = 3
  bias = False

  input_tensor = torch.randn(batch_size, in_features)
  weight = torch.randn(out_features, in_features)
  expected_output = torch.nn.functional.linear(input_tensor, weight)

  qlinear = QLinear(in_features, out_features, bias)
  qlinear.quantize(weight)
  generated_output = qlinear(input_tensor)

  print((expected_output - generated_output).square().mean())