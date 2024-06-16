# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:38:51 2024

@author: Vikram Sandu

Description
===========
Pixel Attention as described in the latent diffusion implementation.
"https://github.com/CompVis/latent-diffusion"

Add-on: Incorporating number of heads information to the existing implementation.

References
==========
1. "https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py"

"""

# (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C) -> 
# (B, H*W, h, hdim) -> (B, h, H*W, hdim) -> (B, H*W, h, hdim) -> (B, H*W, C)
# -> (B, C, H*W) -> (B, C, H, W)

import torch
import torch.nn as nn

class PixelAttention(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 num_heads:int,
                 drop_prob:float = 0.0,
                 skip_connection = False
                 ) -> None:
        super().__init__()
        
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.drop_prob = drop_prob
        self.skip_connection = skip_connection
        
        self.w_q = nn.Conv2d(in_channels,  # Query weights
                             in_channels, 
                             kernel_size=1, 
                             stride=1, 
                             padding=0
                             )
        
        self.w_k = nn.Conv2d(in_channels, # Key weights
                             in_channels, 
                             kernel_size=1, 
                             stride=1, 
                             padding=0
                             )
        
        self.w_v = nn.Conv2d(in_channels, # Value weights
                             in_channels, 
                             kernel_size=1, 
                             stride=1, 
                             padding=0
                             )
        
        self.w_o = nn.Conv2d(in_channels, # Output weights
                             in_channels, 
                             kernel_size=1, 
                             stride=1, 
                             padding=0
                             )
    
    def forward(self, q, k=None, v=None, mask=None):
        
        # If only one input is fed then consider query=key=value
        if (k is None) and (v is None):
            k = q
            v = q
            
        b, c, h, w = q.shape # Batch Size, channels, Height, and Width
        
        query = self.w_q(q) # (B, C, H, W) => (B, C, H, W)
        key = self.w_k(k) # (B, C, H, W) => (B, C, H, W)
        value = self.w_v(v) # (B, C, H, W) => (B, C, H, W)
        
        query = query.view(b, c, h*w).permute(0, 2, 1) # (B, C, H, W) => (B, H*W, C)
        key = key.view(b, c, h*w).permute(0, 2, 1) # (B, C, H, W) => (B, H*W, C)
        value = value.view(b, c, h*w).permute(0, 2, 1) # (B, C, H, W) => (B, H*W, C)
        
        query = query.contiguous().view(b, h*w, self.num_heads, self.head_dim).transpose(1, 2) # (B, H*W, C) => (B, num_heads, H*W, head_dim)
        key = key.contiguous().view(b, h*w, self.num_heads, self.head_dim).transpose(1, 2) # (B, H*W, C) => (B, num_heads, H*W, head_dim)
        value = value.contiguous().view(b, h*w, self.num_heads, self.head_dim).transpose(1, 2) # (B, H*W, C) => (B, num_heads, H*W, head_dim)
        
        # No Dropout during inference.
        use_drop = 0. if not self.training else self.drop_prob
        
        # (B, num_heads, H*W, head_dim) => (B, num_heads, H*W, head_dim)
        context_vec = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=use_drop)
        
        # (B, num_heads, H*W, head_dim) => (B, H*W, num_heads, head_dim) => (B, H*W, C) => (B, C, H*W)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, h*w, c).transpose(1, 2)
        
        # (B, C, H*W) => (B, C, H, W)
        context_vec = context_vec.contiguous().view(b, c, h, w)
        
        if self.skip_connection:
            return q + self.w_o(context_vec)
        
        return self.w_o(context_vec) # (B, C, H, W)

# Test
if __name__ == "__main__":
    
    # Input
    b, c, h, w = 4, 32, 28, 28
    x = torch.randn(b, c, h, w)
    
    # Pixel-Attention
    pa = PixelAttention(32, 4)
    out = pa(x)
    
    print(f'Output shape: {out.shape}')