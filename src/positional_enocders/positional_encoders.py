import torch
from typing import Tuple
import torch.nn as nn
import math


# Sinusoidal positional encoding
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        """Calculates the sinusoidal positional encoding for a given embedding dimension 
        and maximum sequence length.

        Args:
            d_model (int): embedding dimension of the model.
            max_len (int): maximum sequence length for which to calculate positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        for position in range(self.max_len):
            for i in range(0, self.d_model, 2):
                self.pe[position, i] = math.sin(position / (10000 ** (i / self.d_model)))
                self.pe[position, i + 1] = math.cos(position / (10000 ** (i / self.d_model)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the positional embedding to all the batches of x.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: tensor with positional embeddings added of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)


# Learnable positional encoding
class LearnablePE(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        """Calculates the learnable positional encoding for a given embedding dimension 
        and maximum sequence length.

        Args:
            d_model (int): embedding dimension of the model.
            max_len (int): maximum sequence length for which to calculate positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the positional embedding to all the batches of x.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: tensor with positional embeddings added of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)
                

# RoPE positional encoding
class RoPE(nn.Module):
    def __init__(self, d_model: int, rope_base: int = 10000):
        """Calculates the RoPE positional encoding for a given embedding dimension 
        and base.

        Args:
            d_model (int): embedding dimension of the model.
            rope_base (int): base for calculating the RoPE positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.rope_base = rope_base

        self.w = (1 / self.rope_base) ** (torch.arange(0, self.d_model, 2).float() / self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the RoPE positional encoding to the query and key tensors.

        Args:
            Q (torch.Tensor): query tensor of shape (batch_size, seq_len, d_model)
            K (torch.Tensor): key tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tuple[Q, K]: tensors with RoPE positional encodings applied of shape (batch_size, seq_len, d_model)
        """
        seq_len = Q.size(1)
        position = torch.arange(seq_len, device=Q.device).unsqueeze(1)
        self.w = self.w.to(Q.device)
        rotation_angle = position * self.w

        cos = torch.cos(rotation_angle)
        sin = torch.sin(rotation_angle)

        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                rotation_index = i//2

                Q_1 = Q[:, pos, i].clone()
                Q_2 = Q[:, pos, i+1].clone()
                Q[:, pos, i] = Q_1 * cos[pos, rotation_index] - Q_2 * sin[pos, rotation_index]
                Q[:, pos, i+1] = Q_1 * sin[pos, rotation_index] + Q_2 * cos[pos, rotation_index]

                K_1 = K[:, pos, i].clone()
                K_2 = K[:, pos, i+1].clone()
                K[:, pos, i] = K_1 * cos[pos, rotation_index] - K_2 * sin[pos, rotation_index]
                K[:, pos, i+1] = K_1 * sin[pos, rotation_index] + K_2 * cos[pos, rotation_index]

        return Q, K


# ALiBi positional encoding
class ALiBi(nn.Module):
    def __init__(self, n_heads: int):
        """ALiBi positional encoding. Adding bias to attention scores.
        Note:
            Uses a symmetric distance penalty -slope * |i - j|
            instead of the original causal ALiBi bias.
        Args:
            n_heads (int): number of attention heads in the attention mechanism.
        """
        super().__init__()
        self.n_heads = n_heads

    def forward(self, attention_score: torch.Tensor) -> torch.Tensor:
        """Adding ALiBi to attention scores for all the batches and heads.

        Args:
            attention_score (torch.Tensor): Attention scores with shape (batch_size, n_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention scores with ALiBi of shape (batch_size, n_heads, seq_len, seq_len).
        """
        alibi_bias = torch.zeros((attention_score.size(1), attention_score.size(2), attention_score.size(3)),
                                 device=attention_score.device)


        for head_num in range(self.n_heads):
            slope = 2 ** (-(8 / self.n_heads) * (head_num + 1))
            for i in range(alibi_bias.size(1)):
                for j in range(alibi_bias.size(2)):
                    alibi_bias[head_num , i, j] = -slope * abs(j - i)


        return attention_score + alibi_bias.unsqueeze(0)
