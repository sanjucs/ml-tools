# Quantization

The quantization is the process of converting tensors into lower precision. In neural networks, it converts either model parameters or activations into lower precisions such as bfloat16, float16, float8, int8, and int4, which helps to reduce memory and computational requirements.

| datatype | sign | exponent | mantissa | min | max | resolution |
| --- | --- | --- | --- | --- | --- | --- | 
| float32 | 1 | 8 | 23 | 1.17549e-38 | 3.40282e+38 | 1e-06 | 
| float16 | 1 | 5 | 10 | 6.10352e-05 | 65504 | 0.001 | 
| bfloat16 | 1 | 8 | 7 | 1.17549e-38 | 3.38953e+38 | 0.01 |

BFloat16 covers the same dynamic ranges as float32, but precision is the least compared to others since there are fewer numbers of mantissa bits.

## Linear quantization

Linear quantization is one of the techniques used to convert a continuous range of numbers into lower and discrete numbers. e.g., converting from Float32 to INT8. The linear quantization can be defined as

$r = s(q - z)$ where

$r$ = floating point real value

$q$ = interger quantized number

$s$ = quantization scaling factor

$z$ = quantization zero-point

### Calculate scale and zero-point

Consider quantization of numbers from [$r_{min}$, $r_{max}$] to [$q_{min}$, $q_{max}$]. 

$r_{min} = s \cdot (q_{min} - z)$

$r_{max} = s \cdot (q_{max} - z)$

$r_{max} - r_{min} = s \cdot (q_{max} - q_{min}) $

$s = {r_{max} - r_{min} \over q_{max} - q_{min}}$

$z = q_{min} - {r_{min} \over s}$

Code implementation for linear quantization and dequantization can be found [here](/notes/dl/modules/quantization.py)