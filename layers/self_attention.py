# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:38:30 2024

@author: Vikram Sandu

Description
===========
Multi-head-self-attention as described in the paper "Attention is all you need."
"https://arxiv.org/abs/1706.03762"

References
==========

1. "https://www.youtube.com/watch?v=ISNdQcPhsts&t=1918s" by Umar Jamil
2. "https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb"
   Speed Comparison for different implementations of MultiHeadAttention
"""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim:int, 
                 num_heads:int, 
                 drop_prob:float = 0.0 # Dropout Probability
                 ) -> None:
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads"
        
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False) # Query weights
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False) # Key weights
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False) # Value weights
        
        self.w_o = nn.Linear(embed_dim, embed_dim) # Output weights
        
    
        
    def forward(self, q, k, v, mask=None):
        
        query = self.w_q(q) # (B, Seq_len, embed_dim) => (B, Seq_len, embed_dim)
        key = self.w_k(k) # (B, Seq_len, embed_dim) => (B, Seq_len, embed_dim)
        value = self.w_v(v) # (B, Seq_len, embed_dim) => (B, Seq_len, embed_dim)
        
        # (B, Seq_len, embed_dim) => (B, num_heads, Seq_len, head_dim)
        # Transpose => Every head should be able to see whole Sequence
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # No Dropout during inference
        use_dropout = 0. if not self.training else self.drop_prob
        
        # (B, num_heads, Seq_len, head_dim) => (B, num_heads, Seq_len, head_dim)
        context_vector = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=use_dropout)
        
        # (B, num_heads, Seq_len, head_dim) => (B, Seq_len, embed_dim)
        context_vector = context_vector.transpose(1, 2).contiguous().view(context_vector.shape[0], -1, self.num_heads * self.head_dim)
        
        return self.w_o(context_vector) # (B, Seq_len, embed_dim)
        

if __name__ == "__main__":
    
    # Input
    batch_size, seq_len, embed_dim = 4, 8, 512
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Multi-head-attention
    mha = MultiHeadAttention(512, 8)
    out = mha(x, x, x)
    
    print(f'Output shape: {out.shape}')
           