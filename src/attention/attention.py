import torch.nn as nn
import torch
from typing import Tuple
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        """Multi Head Attention mechanism with global, random, and local attention combined.

        Args:
            cfg (Hyperparams): Object of the Hyperparams class in hyperparams.py file.
        """
        super().__init__()

        self.embed_dim = cfg.embedding_size
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.embedding_size // cfg.n_heads

        self.W_q = nn.Linear(cfg.embedding_size, cfg.embedding_size)
        self.W_k = nn.Linear(cfg.embedding_size, cfg.embedding_size)
        self.W_v = nn.Linear(cfg.embedding_size, cfg.embedding_size)

        self.W_o = nn.Linear(cfg.embedding_size, cfg.embedding_size)

        self.dropout = nn.Dropout(cfg.dropout)

        self.cfg = cfg

    def compute_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computer Query, Key, Value matrices.

        Args:
            x (torch.Tensor): input tensor of size (batch_size, seq_len, embed_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns a tuple of Q, K, V
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        return Q, K, V
    
    def qkv_for_heads(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Divide the embed_dim in Q, K, V matrices for each head.

        Args:
            Q (torch.Tensor): Query matrix with shape (batch_size, seq_len, embed_dim)
            K (torch.Tensor): Key matrix with shape (batch_size, seq_len, embed_dim)
            V (torch.Tensor): Value matrix with shape (batch_size, seq_len, embed_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A Tuple of new Q, K, V matrices.
        """
        batch_size, seq_len, embed_dim = Q.shape
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)

        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)

        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)

        return Q, K, V
    
    def compute_sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse attention using local, random, and global patterns.

        This function assumes:
        - A fixed local window size: cfg.local_window
        - A fixed number of random neighbors: cfg.num_random
        - A list of global token indices: cfg.global_indices

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
            K (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len, head_dim)
            V (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_len, head_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_heads, seq_len, head_dim)
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        device = Q.device

        local_window = self.cfg.local_window
        num_random = self.cfg.num_random
        global_indices = self.cfg.global_indices  # list[int]

        output = torch.zeros(batch_size, n_heads, seq_len, head_dim, device=device)

        for i in range(seq_len):
            # build allowed indices for token i
            idx = []

            if i in global_indices:
                idx = [x for x in range(seq_len)]

            else:
                # local window
                start = max(0, i - local_window)
                end = min(seq_len, i + local_window + 1)
                idx.extend(range(start, end))

                # random neighbors
                if num_random > 0:
                    rand = torch.randint(0, seq_len, (num_random,), device=device).tolist()
                    idx.extend(rand)

                # global tokens
                idx.extend(global_indices)

                # remove duplicates while preserving order
                idx = list(dict.fromkeys(idx))

            # select K, V for allowed positions
            K_i = K[:, :, idx, :]  # (B, H, k, D)
            V_i = V[:, :, idx, :]  # (B, H, k, D)

            # query for position i
            Q_i = Q[:, :, i:i+1, :]  # (B, H, 1, D)

            # scaled dot-product attention over allowed positions
            scores = (Q_i @ K_i.transpose(-1, -2)) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)

            out_i = attn @ V_i  # (B, H, 1, D)
            output[:, :, i:i+1, :] = out_i

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi‑head sparse attention to the input sequence.

        Steps:
        1. Compute dense Q, K, V projections.
        2. Split each into multiple heads.
        3. Apply sparse attention (local + random + global patterns).
        4. Concatenate all heads back into a single embedding.
        5. Apply the final output projection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        Q, K, V = self.compute_qkv(x)
        Q, K, V = self.qkv_for_heads(Q, K, V)

        attn_output = self.compute_sparse_attention(Q, K, V)

        # Concat each head attention output
        batch_size, num_heads, seq_len, head_dim = attn_output.shape
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, head_dim * num_heads)

        output = self.W_o(attn_output)

        return(output)
