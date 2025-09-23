import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from abc import ABC, abstractmethod
import math

def sigmoid(v):
    return 1/(1 + torch.exp(-v))


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        return x
    

        
        


class UniformQuant:
    def __init__(self, n_bits:int, symmetric:bool=True):
        self.n_bits =  n_bits
        self.qmax = (2**(n_bits -1))-1
        self.qmin = -(2**(n_bits - 1))
        self.xmin = None
        self.xmax = None
        self.symm = symmetric
        self.zp = None
        self.scale = None
        self.observer = NotImplementedError
        
    def calc_scale(self, x:torch.Tensor):
        self.minmax(x)
        if not self.symm: 
            return torch.tensor((self.xmax - self.xmin) / (self.qmax - self.qmin))
        else:
            return torch.tensor(self.xmax / self.qmax)
    def calc_zero(self):
        if not self.symm:
            quotient = torch.round(self.xmax/self.scale)
            return (self.qmax - quotient).detach().clone()
            # return torch.tensor(self.qmax - (torch.round(torch.tensor(self.xmax/self.scale))))
        else: return torch.tensor(0)
    def quantize(self, x:torch.Tensor):
        self.scale = self.calc_scale(x)
        self.zp = self.calc_zero()
        return torch.clamp(torch.round(x/self.scale) + self.zp, self.qmin, self.qmax)
    def dequantize(self, x_q:torch.Tensor):
        return torch.tensor(self.scale * (x_q - self.zp))
    def minmax(self, x:torch.Tensor):
        if self.xmax is None:
            self.xmax = x.max().item()
        else: self.xmax = max(self.xmax, x.max().item())
        if not self.symm:
            if self.xmin is None:
                self.xmin = x.min().item()
            else: self.xmin = min(self.xmin, x.min().item())
    def returnminmax(self):
        print(f"xmax = {self.xmax} and xmin = {self.xmin}")
    def __str__(self):
        return f"scale = {self.scale} and zp = {self.zp}"
     
class Clamp(UniformQuant):
    def __init__(self, n_bits, symmetric = True, highC=None, lowC=None):
        super().__init__(n_bits, symmetric)
        self.scale = torch.tensor(1.0)
        self.zp = torch.tensor(0.0)
        self.high = highC
        self.low = lowC
    def quantize(self, x):
        self.xmax = x.max().item()
        if self.high is None:
            self.high = 6.0
        if self.low is None:
            self.low = -6.0
        if self.symm:
            self.xmin = - self.xmax
        else: self.xmin = x.min().item()
        
        if self.xmax > self.high and self.xmin < self.low:
            mask = torch.where(x > self.high, 1.0, 0.0) + torch.where(x < self.low, 1.0, 0.0)
            mask = 1 - mask
            # mask = self.xmax < 6.0 + self.xmin > -6.0
            return  x * mask
        else: return x 
        
 
class AdaQuant:
    def __init__(self, n_bits:int, symmetric:bool=True):
        self.n_bits = n_bits
        self.qmax = (2**(n_bits -1))-1
        self.qmin = -(2**(n_bits - 1))
        self.zero_point = 0 if symmetric else NotImplementedError
        pass 
    
    def quantize(self, data):
        pass
    
    def dequantize(self, data):
        pass
    
    def rect_sigmoid(self, v):
        return torch.clamp((sigmoid(v)*1.2) -0.1, 0, 1 )
    
    def forward(self, x):
        x_max = x.max()
        x_min = x.min()
        
        pass
    



        


    
    
    
    
class EditLayer:
    def __init__(self, model:nn.Module, 
                 wtype:Union[UniformQuant, AdaQuant], 
                 atype:Union[UniformQuant, AdaQuant], 
                 w_qparams:dict, 
                 a_qparams:dict,
                 ):
        # super().__init__(qtype, q_params)
        self.model = model
        # self.count = 0
        self.wquant = wtype
        self.aquant = atype
        self.w_qparams = w_qparams
        self.a_qparams = a_qparams
        
    def traverse(self, replace=True):
        change_count = 0
        count = 0
        iter_model = iter(self.model.named_modules())
        if replace:print("Changing to inference mode")
        else: print("Replacing layer in the model...")
        next(iter_model)
        for name, layer in iter_model:
            if not replace:
                if isinstance(layer, QuantConv) or isinstance(layer, QuantConv):
                    layer.calib = False
                    
            else:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight
                    bias_val = layer.bias
                    
                    replacedLayer = QuantLinear(self.wquant(**self.w_qparams), self.aquant(**self.a_qparams), weight=weight, bias=bias_val)
                    self.model.set_submodule(name, replacedLayer, strict=True)
                    change_count += 1
                    print(f"The {name} layer is replaced by {replacedLayer} with {layer.weight.shape} and {layer.bias.shape if layer.bias is not None else None}")
                if isinstance(layer, nn.Conv2d):
                    weight = layer.weight
                    bias_val = layer.bias
                    conv_params = {
                        'in_channels': layer.in_channels, 
                        'out_channels': layer.out_channels,
                        'stride': layer.stride,
                        'padding': layer.padding,
                        'kernel': layer.kernel_size
                    }
                    replacedLayer = QuantConv(self.wquant(**self.w_qparams), self.aquant(**self.a_qparams), weight, bias_val, conv_params)
                    self.model.set_submodule(name, replacedLayer, strict=True)
                    change_count += 1
                    print(f"The {name} layer is replaced by {replacedLayer}")
                    
            count+=1
        if replace: print(f"Total {count} layers. And total {change_count} layers changed.")
        else: print(f"All {count} layers changed to test mode.")
        
    def get_model(self):
        return self.model
    
    def change_blocks_layers(self, block_name=None, module_name=None):
        count = 0
        iter_model = iter(self.model.named_modules())
        if block_name is not None:
            next(iter_model)
            for name, layer in iter_model:
                # for block level quantization
                if block_name in name:
                    if module_name is None:
                        if isinstance(layer, nn.Linear):
                            weight = layer.weight
                            bias_val = layer.bias
                            
                            replacedLayer = QuantLinear(self.wquant(**self.w_qparams), self.aquant(**self.a_qparams), weight=weight, bias=bias_val)
                            self.model.set_submodule(name, replacedLayer, strict=True)
                            count+=1
                        if isinstance(layer, nn.Conv2d):
                            weight = layer.weight
                            bias_val = layer.bias
                            conv_params = {
                                'in_channels': layer.in_channels, 
                                'out_channels': layer.out_channels,
                                'stride': layer.stride,
                                'padding': layer.padding,
                                'kernel': layer.kernel_size
                            }
                            replacedLayer = QuantConv(self.wquant(**self.w_qparams), self.aquant(**self.a_qparams), weight, bias_val, conv_params)
                            self.model.set_submodule(name, replacedLayer, strict=True)
                            count+=1
                    elif module_name in name:
                        if isinstance(layer, nn.Linear):
                            weight = layer.weight
                            bias_val = layer.bias
                            
                            replacedLayer = QuantLinear(self.wquant(**self.w_qparams), self.aquant(**self.a_qparams), weight=weight, bias=bias_val)
                            self.model.set_submodule(name, replacedLayer, strict=True)
                            count+=1
                        if isinstance(layer, nn.Conv2d):
                            weight = layer.weight
                            bias_val = layer.bias
                            conv_params = {
                                'in_channels': layer.in_channels, 
                                'out_channels': layer.out_channels,
                                'stride': layer.stride,
                                'padding': layer.padding,
                                'kernel': layer.kernel_size
                            }
                            replacedLayer = QuantConv(self.wquant(**self.w_qparams), self.aquant(**self.a_qparams), weight, bias_val, conv_params)
                            self.model.set_submodule(name, replacedLayer, strict=True)
                            count+=1
                       
            print(f"Total {count} layers in the {block_name}{'.'+module_name if module_name is not None else ''} replaced.")
                
    

            
class QuantLinear(nn.Module):
    def __init__(self, wtype:Union[UniformQuant, AdaQuant], 
                 atype:Union[UniformQuant, AdaQuant], 
                 weight:torch.Tensor, 
                 bias:torch.Tensor,
                 calib_mode:bool=True):
        super().__init__()
        self.wtype = wtype
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else: 
            self.bias = 0
        self.calib = calib_mode
        self.atype = atype
        self.register_buffer('w_scale', None)
        self.register_buffer('w_zp', None)
        self.register_buffer('a_scale', None)
        self.register_buffer('a_zp', None)
        if calib_mode:
            self.weight = nn.Parameter(self.wtype.quantize(weight))
            self.w_scale = self.wtype.scale
            self.w_zp = self.wtype.zp
    def forward(self, x:torch.Tensor):
        x_q = self.atype.quantize(x)
        if self.calib:
            self.a_scale = self.atype.scale
            self.a_zp = self.atype.zp
        y_ = torch.matmul(x_q - self.atype.zp, self.weight.T - self.wtype.zp)
        deq_y = (self.atype.scale * self.wtype.scale ) * y_
        return deq_y + self.bias      
        

class QuantConv(nn.Module):
    def __init__(self, 
                 wtype:Union[UniformQuant, AdaQuant],
                 atype:Union[UniformQuant, AdaQuant], 
                 weight:torch.Tensor, 
                 bias:torch.Tensor,
                 conv_params:dict,
                 calib_mode:bool=True,
                 ):
        super().__init__()
        # self.wtype = wtype(**wtype_params)
        # self.atype = atype(**atype_params)
        self.wtype = wtype
        self.atype = atype
        self.in_channels = conv_params['in_channels']
        self.out_channels = conv_params['out_channels']
        self.stride = conv_params['stride']
        self.padding = conv_params['padding']
        self.kernel_size = conv_params['kernel']
        
        self.register_buffer('w_scale', None)
        self.register_buffer('w_zp', None)
        self.register_buffer('a_scale', None)
        self.register_buffer('a_zp', None)
        self.calib = calib_mode
        if calib_mode:
            self.weight = nn.Parameter(self.wtype.quantize(weight))
            self.w_scale = self.wtype.scale
            self.w_zp = self.wtype.zp
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else: self.bias = 0
        
    def _calcOutputSize(self, h_in, w_in):
        h_out = math.floor(((h_in + (2 * self.padding[0]) -self.kernel_size[0]  ) / self.stride[0]) + 1)
        w_out = math.floor(((w_in + (2 * self.padding[1]) -self.kernel_size[1] ) / self.stride[1]) + 1)
        return h_out, w_out
    def forward(self, x:torch.Tensor):
        _, _, h_in, w_in = x.shape 
        h_out, w_out = self._calcOutputSize(h_in, w_in)
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x_unf = self.atype.quantize(x_unf)
        if self.calib:
            self.a_scale = self.atype.scale
            self.a_zp = self.atype.zp
        out_unf = torch.matmul((x_unf.transpose(1, 2) - self.atype.zp), self.weight.view(self.weight.size(0), -1).t() - self.wtype.zp) 
        out_unf = ((self.atype.scale * self.wtype.scale) * out_unf) + self.bias
        out_unf_dq = out_unf.transpose(2, 1) 
        out_fold = nn.functional.fold(out_unf_dq, output_size=(h_out, w_out), kernel_size=(1, 1)) 
        return out_fold 
    
    
    
