import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Union
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

# class BaseQuantBlock(nn.Module):
#     @abstractmethod
#     def __init__(self, q_type):
#         super().__init__()
#         pass
  
  
  
  
# class TemporalInfoQuantBlock(nn.Module):
#     def __init__(self,  time_embed:TimeStepEmbedding):
#         super().__init__()
        
        
class Timesteps(nn.Module):
    pass    
        
class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class ResnetBlock2D(nn.Module):
    pass

class Attention(nn.Module):
    pass

class GEGLU(nn.Module):
    pass

class FeedForward(nn.Module):
    pass


class BasicTransformerBlock(nn.Module):
    pass

class TransformerBlock2DModel(nn.Module):
    pass

class Downsample2D(nn.Module):
    pass

class Unpsample2D(nn.Module):
    pass

class DownBlock2D(nn.Module):
    pass

class CrossAttnDownBlock2D(nn.Module):
    pass

class CrossAttnUpBlock2D(nn.Module):
    pass

class UpBlock2D(nn.Module):
    pass

class UNetMidBlock2DCrossAttn(nn.Module):
    pass

class UNet2DConditionModel(ModelMixin, ConfigMixin):
    pass

        