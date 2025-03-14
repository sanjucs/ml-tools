# Quantization

The quantization is the process of converting tensors into lower precision. In neural networks, it converts either model parameters or activations into lower precisions such as bfloat16, float16, float8, int8, and int4, which helps to reduce memory and computational requirements.

| datatype | sign | exponent | mantissa | min | max | resolution |
| --- | --- | --- | --- | --- | --- | --- | 
| float32 | 1 | 8 | 23 | 1.17549e-38 | 3.40282e+38 | 1e-06 | 
| float16 | 1 | 5 | 10 | 6.10352e-05 | 65504 | 0.001 | 
| bfloat16 | 1 | 8 | 7 | 1.17549e-38 | 3.38953e+38 | 0.01 |

BFloat16 covers the same dynamic range as float32, but it has the least precision when compared to the others since there are fewer numbers of mantissa bits.

## Linear quantization

Linear quantization is one of the techniques used to convert a continuous range of numbers into lower and discrete numbers. e.g., converting from Float32 to INT8. Linear quantization can be defined as

$$r = s \cdot (q - z)$$

where

$r$ : floating point real value

$q$ : interger quantized number

$s$ : quantization scaling factor

$z$ : quantization zero-point

### Calculate scale and zero-point

Consider quantization of numbers from $[r_{min}, r_{max}]$ to $[q_{min}, q_{max}]$.

$$r_{min} = s \cdot (q_{min} - z)$$

$$r_{max} = s \cdot (q_{max} - z)$$

$$r_{max} - r_{min} = s \cdot (q_{max} - q_{min})$$

$$s = {r_{max} - r_{min} \over q_{max} - q_{min}}$$

$$z = q_{min} - {r_{min} \over s}$$

There are two types of linear quantization: symmetric and asymmetric. In the former, the absolute maximum of the tensor ($r_{max}$) defines the range as $[-r_{max}, r_{max}]$, whereas the latter considers the min and max of the tensor $[r_{min}, r_{max}]$. Code implementation for linear quantization and dequantization can be found [here](/tools/quantization/quantization.py). Similarly, there are per-channel quantization, per-tensor quantization, and per-group quantization, which quantize per tensor level, channel level, and group level, respectively.

---

## Reference
Courses
* https://learn.deeplearning.ai/courses/quantization-fundamentals/
* https://www.deeplearning.ai/short-courses/quantization-in-depth/