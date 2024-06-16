# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:54:01 2024

@author: Vikram Sandu

Description
===========
Pytorch implementation of the VQ-VAE as described in the latent diffusion implementation. "https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py"

Note: This code is ported from "https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py" and modified further for simplicity.

References
==========
1. Latent Diffusion Autoencoder. https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
2. Variational Autoencoder by Umar Jamil. "https://www.youtube.com/watch?v=iwEzwTTalbg"
3. Implementing Variational Auto Encoder from Scratch in Pytorch by ExplainingAI. "https://www.youtube.com/watch?v=pEsC0Vcjc7c"

"""

# Imports
import math
import yaml
import torch
import torch.nn as nn
from layers.pixel_attention import PixelAttention

######################## Utility Modules ########################

class NormActConv(nn.Module):
    """
    Perform GroupNorm, Activation (SiLU), and Conv operations.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int = 3,
                 num_groups:int = 32,
                 stride:int = 1,
                 use_norm:bool = True, 
                 use_act:bool = True, 
                 use_conv:bool = True
                 ) -> None:
        super(NormActConv, self).__init__()
        
        self.groupnorm = nn.GroupNorm(num_groups, in_channels) if use_norm else nn.Identity()
        self.act = nn.SiLU() if use_act else nn.Identity()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size = kernel_size, 
                              padding=(kernel_size-1)//2, 
                              stride=stride
                              ) if use_conv else nn.Identity()
        
    def forward(self, x):
        
        x = self.groupnorm(x)
        x = self.act(x)
        x = self.conv(x)
        
        return x



class ResnetBlock(nn.Module):
    """
    Resnet Block used in Latent Diffusion implementation.
    This is a stack of 2 NormActConv blocks with Residual.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int = None,
                 num_layers:int = 2, # Number of NormActConv blocks
                 ) -> None:
        super(ResnetBlock, self).__init__()
        
        out_channels = in_channels if out_channels is None else out_channels
        
        self.blocks = nn.Sequential(*[
            NormActConv(
                in_channels if i==0 else out_channels, 
                out_channels
                ) for i in range(num_layers)
            ])
        
        
        self.conv_shortcut = nn.Conv2d(in_channels, 
                                       out_channels, 
                                       kernel_size=1
                                       ) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        
        res_input = x
        x = self.blocks(x)
        x = x + self.conv_shortcut(res_input) # Residual connection
        
        return x
    
    
class Downsample(nn.Module):
    """
    Downsample the image by a factor of k.
    Add-ons: Allows to concat features from both Conv and AvgPooling.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int = None, 
                 k:int = 2,
                 use_conv:bool=True, 
                 use_pool:bool=True
                 ) -> None:
        super(Downsample, self).__init__()
        
        self.use_conv = use_conv
        self.use_pool = use_pool
        out_channels = in_channels if out_channels is None else out_channels
        
        self.down_conv = nn.Conv2d(in_channels, 
                                   out_channels//2 if self.use_pool else out_channels, 
                                   kernel_size = 4, # For symmetry
                                   stride=k,
                                   padding=1
                                   ) if self.use_conv else nn.Identity()
        
        self.down_pool = nn.Sequential(
            nn.AvgPool2d(k, k), 
            nn.Conv2d(in_channels, 
                      out_channels//2 if self.use_conv else out_channels, 
                      kernel_size = 1
                      )
            ) if self.use_pool else nn.Identity()
        
    def forward(self, x):
        
        if not self.use_conv:
                return self.down_pool(x)
            
        if not self.use_pool:
            return self.down_conv(x)
        
        return torch.cat([self.down_conv(x), self.down_pool(x)], dim=1)
            

class Upsample(nn.Module):
    """
    Upsample the image by a factor of k.
    Add-ons: Allows to concat features from both ConvTranspose2D and nn.Upsample.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int = None, 
                 k:int=2, 
                 use_conv:bool=True, 
                 use_upsample:bool=True
                 ) -> None:
        super(Upsample, self).__init__()
        
        out_channels = in_channels if out_channels is None else out_channels
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        
        self.up_conv = nn.ConvTranspose2d(in_channels,
                                          out_channels//2 if self.use_upsample else out_channels, 
                                          kernel_size=4, 
                                          stride=k, 
                                          padding=1 # Need to fix for k > 2 
                                          ) if self.use_conv else nn.Identity()
        
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=k, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 
                      out_channels//2 if self.use_conv else out_channels, 
                      kernel_size=1
                      )
            ) if self.use_upsample else nn.Identity()
        
    def forward(self, x):
        
        if not self.use_conv:
            return self.up_sample(x)
        
        if not self.use_upsample:
            return self.up_conv(x)
        
        return torch.cat([self.up_conv(x), self.up_sample(x)], dim=1)
    
    
###############################################################


######################## Encoder ##############################

class Encoder(nn.Module):
    """
    VQ-VAE Model Encoder. Encode/Compress the Image to Latent space.
    Later Diffusion Model will be trained on this Encoded Image.
    """
    
    def __init__(self, 
                 config # Configuration file.
                 ):
        super(Encoder, self).__init__()
        
        self.in_resolution = config['in_resolution']
        self.out_resolution = config['out_resolution']
        self.in_channels = config['in_channels']
        self.ch = config['ch']
        self.out_channels = config['out_channels']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolution = config['attn_resolution']
        self.max_ch_multiplier = config['max_ch_multiplier']
        self.num_heads = config['num_heads']
        self.num_downsamples = int(math.log2(self.in_resolution//self.out_resolution))
        
        
        # Input Convolution
        self.conv_in = nn.Conv2d(self.in_channels, 
                                 self.ch, 
                                 kernel_size=3, 
                                 padding=1
                                 )
        
        # Downsampling
        self.down = []
        curr_res = self.in_resolution
        for i_level in range(self.num_downsamples):
            block_in = self.ch * min(self.max_ch_multiplier, (i_level + 1))
            block_out = self.ch * min(self.max_ch_multiplier, (i_level + 2))
            
            # ResBlocks
            self.down += [ResnetBlock(block_in if index == 0 else block_out, 
                                      block_out
                                      ) for index in range(self.num_res_blocks)
                          ]
            
            # PixelAttention if Resolution is under self.attn_resolution
            if curr_res <= self.attn_resolution:
                self.down.append(PixelAttention(block_out, self.num_heads))
                
            # Downsample
            self.down.append(Downsample(block_out, block_out))
            
            # Update curresnt resolution
            curr_res = curr_res // 2
            
        # No Downsampling in last set of blocks.
        self.down += [ResnetBlock(block_out) for index in range(self.num_res_blocks)]
        
        # Sequential Unpacking
        self.down = nn.Sequential(*self.down)
        
        # Middle Block (refinement of downsample features)
        self.mid = nn.Sequential(
            ResnetBlock(block_out), 
            PixelAttention(block_out, self.num_heads), 
            ResnetBlock(block_out)
            )
        
        # Output Convolution
        self.conv_out = NormActConv(block_out, 
                                    self.out_channels, 
                                    kernel_size=3
                                    )
        
    
    def forward(self, x):
        
        x = self.conv_in(x)
        x = self.down(x)
        x = self.mid(x)
        x = self.conv_out(x)
        
        return x


###############################################################

######################## Decoder ##############################


###############################################################

# Test
if __name__ == "__main__":
    
    # Load Config
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
   
    # Initialize the Model
    encoder_config = config['encoder']
    encoder = Encoder(encoder_config)
    
    # Initialize Input
    batch_size = 4
    ch = encoder_config['in_channels'] 
    res = encoder_config['in_resolution']
    
    x = torch.randn(batch_size, ch, res, res)
    print(encoder(x).shape)
        