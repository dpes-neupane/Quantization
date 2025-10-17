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
    def quantize(self, x:torch.Tensor, calib_mode=True):
        if calib_mode:
            self.scale = self.calc_scale(x)
            self.zp = self.calc_zero()
        return torch.clamp(torch.round(x/self.scale) + self.zp, self.qmin, self.qmax)
    def dequantize(self, x_q:torch.Tensor):
        if isinstance(x_q, torch.Tensor):
            return self.scale * (x_q - self.zp)
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
        if not self.calib: 
            self.atype.scale = self.a_scale
            self.atype.zp = self.a_zp
            x_q = self.atype.quantize(x, calib_mode=False)
        else:
            x_q = self.atype.quantize(x)
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
        else: self.bias = None

        
    def _calcOutputSize(self, h_in, w_in):
        h_out = math.floor(((h_in + (2 * self.padding[0]) -self.kernel_size[0]  ) / self.stride[0]) + 1)
        w_out = math.floor(((w_in + (2 * self.padding[1]) -self.kernel_size[1] ) / self.stride[1]) + 1)
        return h_out, w_out
    def forward(self, x:torch.Tensor):
        _, _, h_in, w_in = x.shape 
        h_out, w_out = self._calcOutputSize(h_in, w_in)
        if not self.calib: 
            self.atype.scale = self.a_scale
            self.atype.zp = self.a_zp
            x_q = self.atype.quantize(x, calib_mode=False)
        else:
            x_q = self.atype.quantize(x)
            self.a_scale = self.atype.scale
            self.a_zp = self.atype.zp
        x_q = x_q - self.a_zp
        weight = self.weight - self.w_zp
        out_q = F.conv2d(x_q , weight, stride=self.stride, padding=self.padding)
        out_deq = (self.a_scale * self.w_scale) * out_q 
        if self.bias is not None:
            bias = torch.repeat_interleave(self.bias, h_out * w_out).view(self.weight.shape[0], h_out, w_out)
            out_deq = out_deq + bias
        return out_deq 
    
    
    
class QuantLinearCH(QuantLinear):
    
    def __init__(self,  ch_groups=16, **params):
        super().__init__(**params)
        self.ch_groups = ch_groups
        self.register_buffer('min_in_ch', None)
        self.register_buffer('max_in_ch', None)
        self.register_buffer('mean_x', None)
        self.atypes = []
        
        for _ in range(self.ch_groups):
            if isinstance(params['atype'], UniformQuant):
                self.atypes.append(UniformQuant(params['atype'].n_bits, params['atype'].symm))
            else: NotImplementedError
            
    def sort_channel_by_range(self, x:torch.Tensor):
            # min_x = torch.min(x, dim=1)[0]
            b, ch = x.shape[0], x.shape[1]
            if self.calib:
                min_x = torch.min(x.view(1, b * ch, x.shape[2]), dim=1)[0]
                max_x = torch.max(x.view(1, b * ch, x.shape[2]), dim=1)[0]
                self.minmaxing(max_x, min_x)
                self.mean_x = ((self.max_in_ch + self.min_in_ch) / 2).unsqueeze(1)
            x_prime = x - self.mean_x
            act_range = abs(self.max_in_ch - self.min_in_ch)
            #sorting channels using activation range
            return x_prime, torch.argsort(act_range)
        
    def minmaxing(self, max_x_new:torch.Tensor, min_x_new:torch.Tensor):
        if self.max_in_ch is not None:
            bit_mask = torch.gt(self.max_in_ch, max_x_new)
            # false_indices = torch.eq(bit_mask, 0)
            indices = (bit_mask==0)
            self.max_in_ch[indices] = max_x_new[indices]
        else: 
            self.max_in_ch = max_x_new
        if self.min_in_ch is not None:
            bit_mask = torch.lt(self.min_in_ch, min_x_new)
            indices = (bit_mask==0)
            self.min_in_ch[indices] = min_x_new[indices] 
        else:
            self.min_in_ch = min_x_new
         
    
    def sub_matmul(self, x:torch.Tensor, sorted_range_indices:torch.Tensor):
        #group
        b, m, ch = x.shape
        ch, n = self.weight.shape
        b_ = torch.arange(b).reshape(-1, 1)
        
        output = torch.zeros(b, m, n)
        if self.calib:
            self.a_scale = torch.zeros(self.ch_groups)
            self.a_zp = torch.zeros(self.ch_groups)
        per_group_channels = math.ceil(ch / self.ch_groups)
        for i in range(0, ch, per_group_channels):
        #quantize the submatrices of x
            x_sub = x[b_, :, sorted_range_indices[:, i:i+per_group_channels]].transpose(-1, -2)
            if self.calib:
                x_sub_q = self.atypes[i//per_group_channels].quantize(x_sub)
                self.a_scale[i//per_group_channels] = self.atypes[i//per_group_channels].scale
                self.a_zp[i//per_group_channels] = self.atypes[i//per_group_channels].zp
            else:
                self.atypes[i//per_group_channels].scale = self.a_scale[i//per_group_channels] 
                self.atypes[i//per_group_channels].zp = self.a_zp[i//per_group_channels]
                x_sub_q = self.atypes[i//per_group_channels].quantize(x_sub)
            sub_matmul = torch.matmul(x_sub_q - self.atypes[i//per_group_channels].zp, self.weight.T[sorted_range_indices[:, i:i+per_group_channels], :] - self.w_zp)
            output += ((self.atypes[i//per_group_channels].scale * self.wtype.scale) * sub_matmul)
        return output 
                   
    def forward(self, x:torch.Tensor):
        x_p, sorted_indices = self.sort_channel_by_range(x)
        weight_deq = self.wtype.dequantize(self.weight.detach().clone().requires_grad_(False))
        return self.sub_matmul(x_p, sorted_indices) + ((self.mean_x @ weight_deq.T) + self.bias)
    
    
class QuantConvCH(QuantConv):
    def __init__(self, ch_groups=16, **params):
        super().__init__(**params)
        self.atypes = []
        self.ch_groups = ch_groups
        self.register_buffer('min_in_ch', None)
        self.register_buffer('max_in_ch', None)
        self.register_buffer('sorted_ch_indices', None)
        self.register_buffer('mean_x', None)
        for i in range(self.ch_groups):
            if isinstance(params['atype'], UniformQuant):
                self.atypes.append(UniformQuant(params['atype'].n_bits, params['atype'].symm))
            else: NotImplementedError
            
    def sort_channel_by_range(self, x:torch.Tensor, b:int, ch:int, h:int, w:int):
        if self.calib:
            min_x = torch.min(x.view(1,  ch, b * h * w), dim=-1)[0]
            max_x = torch.max(x.view(1, ch, b * h * w), dim=-1)[0]
            self.minmaxing(max_x, min_x)
            self.mean_x = ((self.max_in_ch + self.min_in_ch) / 2).unsqueeze(2).unsqueeze(2)
        x_prime = x - self.mean_x
        act_range = abs(self.max_in_ch - self.min_in_ch)
        #sorting channels using activation range
        return x_prime, torch.argsort(act_range)
        
    def minmaxing(self, max_x_new:torch.Tensor, min_x_new:torch.Tensor):
        if self.max_in_ch is not None:
            bit_mask = torch.gt(self.max_in_ch, max_x_new)
            indices = (bit_mask==0)
            self.max_in_ch[indices] = max_x_new[indices]
        else: 
            self.max_in_ch = max_x_new
        if self.min_in_ch is not None:
            bit_mask = torch.lt(self.min_in_ch, min_x_new)
            indices = (bit_mask==0)
            self.min_in_ch[indices] = min_x_new[indices] 
        else:
            self.min_in_ch = min_x_new
            
    def sub_convolution(self, x:torch.Tensor, sorted_range_indices:torch.Tensor, 
                 h_out:int, w_out:int, b:int, ch:int, h:int, w:int, out_ch:int, in_ch:int, k1:int, k2:int):
        b_ = torch.arange(b).reshape(-1, 1)
        output = torch.zeros(b, out_ch, h_out, w_out)
        if self.calib:
            self.a_scale = torch.zeros(self.ch_groups)
            self.a_zp = torch.zeros(self.ch_groups)
        per_group_channels = math.ceil(ch / self.ch_groups)
        for i in range(0, ch, per_group_channels):
            x_sub = x[b_, sorted_range_indices[:, i:i+per_group_channels], :, :]
            if self.calib:
                x_sub_q = self.atypes[i//per_group_channels].quantize(x_sub)
                self.a_scale[i//per_group_channels] = self.atypes[i//per_group_channels].scale
                self.a_zp[i//per_group_channels] = self.atypes[i//per_group_channels].zp
            else:
                self.atypes[i//per_group_channels].scale = self.a_scale[i//per_group_channels] 
                self.atypes[i//per_group_channels].zp = self.a_zp[i//per_group_channels]
                x_sub_q = self.atypes[i//per_group_channels].quantize(x_sub)
            x_sub_q = x_sub_q - self.atypes[i//per_group_channels].zp
            weight = self.weight.transpose(0, 1)[sorted_range_indices.squeeze(0)[i:i+per_group_channels], :, :, :].transpose(0, 1) - self.w_zp
            sub_conv = F.conv2d(x_sub_q, weight, stride=self.stride, padding=self.padding)
            out_deq = (self.atypes[i//per_group_channels].scale * self.w_scale) * sub_conv 
            output = output + out_deq
        return output
    
    def forward(self, x:torch.Tensor):
        b, ch, h, w = x.shape
        out_ch, in_ch, k1, k2 = self.weight.shape
        x_p, sorted_indices = self.sort_channel_by_range(x, b, ch, h, w)
        weight_deq = self.wtype.dequantize(self.weight.detach().clone().requires_grad_(False))
        h_out, w_out = super()._calcOutputSize(h, w)
        residual_x = torch.repeat_interleave(self.mean_x.view(ch),  h * w).view(1, ch, h, w)
        residual = F.conv2d(residual_x, weight_deq, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias = torch.repeat_interleave(self.bias, h_out * w_out).view(self.weight.shape[0], h_out, w_out)
            res_bias = residual + bias
            return self.sub_convolution(x_p, sorted_indices, h_out, w_out, b, ch, h, w, out_ch, in_ch, k1, k2) + res_bias
        conv = self.sub_convolution(x_p, sorted_indices, h_out, w_out, b, ch, h, w, out_ch, in_ch, k1, k2)
        return conv + residual
        
        
