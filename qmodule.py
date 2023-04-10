import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F

# S = (r_max - r_min) / (q_max - q_min)
# Z = round(q_max - r_max / S)


def calScaleZeroPoints(min_val, max_val, num_bits=8):
    q_min = 0
    q_max = 2 ** num_bits - 1
    S = float((max_val - min_val) / (q_max - q_min))
    Z = round(q_max - max_val / S)
    if Z > q_max:
        Z = q_max
    elif Z < q_min:
        Z = q_min

    Z = int(Z)
    
    return S, Z

def quantize_tensor(x, S, Z, num_bits=8, signed=False):
    if signed:
        q_min = -2 ** (num_bits - 1)
        q_max = 2 ** (num_bits - 1) - 1

    else:
        q_min = 0
        q_max = 2 ** num_bits - 1

    q_x = x / S + Z
    q_x.clamp_(q_min, q_max).round_()

    return q_x.float()  # return value should be transformede into float-point form since PyTorch does not support integer computing

def dequantize_tensor(q_x, S, Z):
    return S * (q_x - Z)



class QParam:
    def __init__(self, num_bits) -> None:
        self.num_bits = num_bits
        self.S = None
        self.Z = None
        self.max_val = None
        self.min_val = None

    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        
        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()

        self.S, self.Z = calScaleZeroPoints(self.min_val, self.max_val, self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.S, self.Z, self.num_bits)
    
    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.S, self.Z)



class QModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def quantize_tensor(self, x):
        raise NotImplementedError('quantize_tensor is not implemented yet!')
    
