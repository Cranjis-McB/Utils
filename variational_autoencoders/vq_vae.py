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
4. Vector Quantizer by taming-transformers. "https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"
"""

# Imports
import math
import yaml
import torch
import torch.nn as nn
from einops import rearrange
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
        self.z_channels = config['z_channels']
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
        for i_level in range(self.num_downsamples + 1):
            block_in = self.ch * min(self.max_ch_multiplier, (i_level + 1))
            block_out = self.ch * min(self.max_ch_multiplier, (i_level + 2))
            
            # ResBlocks + PixelAttention
            for i in range(self.num_res_blocks):
                
                # ResnetBlock
                self.down.append(
                    ResnetBlock(block_in if i == 0 else block_out, 
                                block_out
                               )
                )
                
               # PixelAttention if Resolution is under self.attn_resolution
                if curr_res <= self.attn_resolution:
                    self.down.append(PixelAttention(block_out, self.num_heads, skip_connection=True))
            
            # Downsample except last
            if i_level != self.num_downsamples:
                self.down.append(Downsample(block_out, block_out))
            
                # Update current resolution
                curr_res = curr_res // 2
        
        # Sequential Unpacking
        self.down = nn.Sequential(*self.down)
        
        # Middle Block (refinement of downsample features)
        self.mid = nn.Sequential(
            ResnetBlock(block_out), 
            PixelAttention(block_out, self.num_heads, skip_connection=True), 
            ResnetBlock(block_out)
            )
        
        # Output Convolution
        self.conv_out = NormActConv(block_out, 
                                    self.z_channels, 
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

class Decoder(nn.Module):
    """
    VQ-VAE Decoder. Decode/Decompress the Latent Image to Original Image.
    -- Inverse of Encoder Architecture
    """
    
    def __init__(self, 
                 config # Configuration
                 ):
        super(Decoder, self).__init__()
        
        self.in_resolution = config['in_resolution']
        self.out_resolution = config['out_resolution']
        self.in_channels = config['in_channels']
        self.ch = config['ch']
        self.out_channels = config['out_channels']
        self.z_channels = config['z_channels']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolution = config['attn_resolution']
        self.max_ch_multiplier = config['max_ch_multiplier']
        self.num_heads = config['num_heads']
        self.num_upsamples = int(math.log2(self.in_resolution//self.out_resolution))
        
        
        block_in = self.ch * self.max_ch_multiplier
        curr_res = self.out_resolution
        
        # Input Convolution
        self.conv_in = nn.Conv2d(self.z_channels, 
                                 block_in, 
                                 kernel_size=3, 
                                 padding=1
                                 )
        
        # Middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in), 
            PixelAttention(block_in, self.num_heads, skip_connection=True), 
            ResnetBlock(block_in)
            )
        
        # Upsampling
        self.up = []
        
        for i_level in reversed(range(self.num_upsamples + 1)):
            
            # Output channel
            block_out = self.ch * min(self.max_ch_multiplier, i_level + 1)
            
            # Upsample except last
            if i_level != self.num_upsamples:
                self.up.append(Upsample(block_in, block_in))
                
                # Update current resolution
                curr_res = curr_res * 2
            
            # ResBlocks + PixelAttention
            for i in range(self.num_res_blocks):
                
                # ResnetBlock
                self.up.append(
                    ResnetBlock(block_in if i == 0 else block_out, 
                                block_out
                               )
                )
                
               # PixelAttention if Resolution is under self.attn_resolution
                if curr_res <= self.attn_resolution:
                    self.up.append(PixelAttention(block_out, self.num_heads, skip_connection=True))
                    
            # Update Input channels                    
            block_in = block_out
        
        # Sequential Unpacking
        self.up = nn.Sequential(*self.up)
        
        # Output Convolution
        self.conv_out = NormActConv(block_out, 
                                    self.out_channels, 
                                    kernel_size=3,
                                    use_act=False
                                    )

        
        
    
    def forward(self, x):
        
        x = self.conv_in(x)
        x = self.mid(x)
        x = self.up(x)
        x = self.conv_out(x)
        
        return x

###############################################################

#################### Vector Quantizer #########################

class VectorQuantizer(nn.Module):
    """
    Quantize the Encoder output according to the codebook.
    """
    def __init__(self, 
                 config
                 ):
        super(VectorQuantizer, self).__init__()
        
        
        self.num_embed = config['num_embed']
        self.embed_dim = config['embed_dim']
        self.beta = config['beta']
        
        # Codebook params
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.codebook.weight.data.uniform_(-1.0/self.num_embed, 1.0/self.num_embed)
    
    def forward(self, x):
        
        # (B, self.embed_dim, H, W) => (B, H, W, self.embed_dim)
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        
        # (B, H, W, C) => (BHW, self.embed_dim)
        x_flattend = x.view(-1, self.embed_dim)
        
        # Distances between x and codebook. x^2 + codebook^2 - 2 * x * codebook
        # (BHW, self.num_embed)
        d = torch.sum(x_flattend ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=1) - 2 * \
            torch.einsum('be, en -> bn', x_flattend, rearrange(self.codebook.weight, 'n e -> e n'))
        
        # Get nearest codebook indices to the encoder output vectors
        min_encoding_indices = torch.argmin(d, dim=1) # (BHW, )
        
        # Get Quantized Output
        # (BHW, self.embed_dim) => (B, H, W, self.embed_dim) 
        x_q = self.codebook(min_encoding_indices).view(x.shape)
        
        # Compute Loss
        loss = self.beta * torch.mean((x_q.detach() - x)**2) + \
               torch.mean((x_q - x.detach())**2)
        
        # Preserve Gradients
        x_q = x + (x_q - x).detach()
        
        # Reshape back to the original input shape
        # (B, H, W, self.embed_dim) => (B, self.embed_dim, H, W)
        x_q = rearrange(x, 'b h w c -> b c h w')
        
        return x_q, loss

###############################################################


####################### VQ-VAE ################################

class VQVAE(nn.Module):
    """
    VQ-VAE Model as implemented in Latent Diffusion Implementation.
    """
    def __init__(self, 
                 config
                 ):
        super(VQVAE, self).__init__()
        
        self.z_channels = config['z_channels']
        self.embed_dim = config['embed_dim']
        
        # Initialize Encoder
        self.encoder = Encoder(config)
        
        # Initialize Vector Quantizer + Pre and Post Conv
        self.quantizer = VectorQuantizer(config)
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.z_channels, kernel_size=1) 
        
        # Initialize Decoder
        self.decoder = Decoder(config)
        
    
    def encode(self, x):
        
        x = self.encoder(x)
        x = self.pre_quant_conv(x)
        x, codebook_loss = self.quantizer(x)
        
        return x, codebook_loss
    
    def decode(self, x):
        
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        
        return x
    
    def forward(self, x):
        
        x, codebook_loss = self.encode(x)
        x = self.decode(x)
        
        return x, codebook_loss

###############################################################

# Test
if __name__ == "__main__":
    
    # Load Config
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
   
    # Initialize the Model
    model_config = config['model']
    vqvae = VQVAE(model_config)
    
    # Initialize Input
    batch_size = 4
    ch = model_config['in_channels'] 
    res = model_config['in_resolution']
    
    # VQ-VAE Model
    x = torch.randn(batch_size, ch, res, res)
    out, cb_loss = vqvae(x)
    print(f'VQ-VAE Output shape: {out.shape}')
    print(f'Codebook Loss: {cb_loss}')
    