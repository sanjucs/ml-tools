# Quantization

The quantization is the process of converting tensors into lower precision. In neural networks, it converts either model parameters or activations into lower precisions such as bfloat16, float16, float8, int8, and int4.

| datatype | # sign | exponent | mantissa | min | max | resolution |
| --- | --- | --- | --- | --- | --- | --- | 
| float32 | 1 | 8 | 23 | 1.17549e-38 | 3.40282e+38 | 1e-06 | 
| float16 | 1 | 5 | 10 | 6.10352e-05 | 65504 | 0.001 | 
| bfloat16 | 1 | 8 | 7 | 1.17549e-38 | 3.38953e+38 | 0.01 |

BFloat16 covers the same dynamic ranges as float32, but precision is the least compared to others since there are fewer numbers of mantissa bits.