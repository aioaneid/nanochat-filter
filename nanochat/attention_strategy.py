from nanochat.gpt import apply_rotary_emb, norm, uniform_normal_std_equalizer
from nanochat.gpt_config import GPTConfig
from abc import ABC, abstractmethod
import torch
from torch import nn


class AttentionStrategy(ABC, nn.Module):
    """
    Abstract Base Strategy.
    Responsibilities:
    1. Define input projection weights (c_q, c_k, c_v).
    2. Define factorization weights (if any).
    3. In forward(): Project inputs, apply RoPE, apply Chain (if any).
    4. Return (q, k, v) ready for SDPA.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim()
        assert self.n_head * self.head_dim == self.n_embd, (
            "n_embd must be divisible by n_head"
        )

    @abstractmethod
    def init_weights(self):
        """
        Initialize non-linear weights (projections, factorization, etc).
        Called from the parent GPT model.
        """
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, T, C)
            cos_sin: Rotary embeddings tuple
        Returns:
            q: (B, H, T, D)
            k: (B, H_kv, T, D) - Note: H_kv might match H if expanded
            v: (B, H_kv, T, D)
        """
        pass

    @abstractmethod
    def get_forward_flops(self, B, T):
        """
        Calculates the exact floating point operations for the Attention mechanism
        (excluding linear projections, which are counted in the block params).

        This accounts for:
        1. The interaction score calculation (N^2 term)
        2. The value aggregation (N^2 term)
        3. Any intermediate chain calculations (Linear term)
        """
        pass


class StandardAttention(AttentionStrategy):
    def __init__(self, config):
        super().__init__(config)
        # Standard Projections
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # arrgh(self.c_q, self.c_k, self.c_v)

    def init_weights(self):
        s = uniform_normal_std_equalizer(self.config.n_embd)
        torch.nn.init.uniform_(self.c_q.weight, -s, s) # weights use Uniform to avoid outliers
        torch.nn.init.uniform_(self.c_k.weight, -s, s)
        torch.nn.init.uniform_(self.c_v.weight, -s, s)
        

    def forward(self, x, cos_sin):
        B, T, _ = x.size()

        # 1. Project
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        # logger.info("a q: %s", q)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        # logger.info("b k: %s", k)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # logger.info("c v: %s", v)

        # 2. RoPE
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 3. Norm
        q, k = norm(q), norm(k)

        # 4. Transpose to (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # logger.info("d q: %s", q)
        # logger.info("e k: %s", k)
        # logger.info("f v: %s", v)

        return q, k, v

    def get_forward_flops(self, B, T):
        # 1. Q @ K.T -> (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        # Cost: 2 * B * H * T^2 * D
        qk_flops = 2 * B * self.n_head * (T**2) * self.head_dim

        # 2. Attn @ V -> (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        # Cost: 2 * B * H * T^2 * D
        # Note: We use n_head here because the result is expanded to n_head
        # even if V uses n_kv_head (due to broadcasting/GQA).
        av_flops = 2 * B * self.n_head * (T**2) * self.head_dim

        return qk_flops + av_flops


class DelegatingKeyStrategy(ABC, nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, x, k):
        """
        Given input x [B, T, n_embd] and keys k [B, T, H_kv, D],
        return gated keys: [B, T, H_kv, D] or [B, T, H, D]
        """
        pass

    @abstractmethod
    def get_forward_flops(self, B, T) -> int:
        """Return approximate FLOPs of the gating computation."""
        pass


class IdentityGate(DelegatingKeyStrategy):
    def __init__(self, config: GPTConfig):
        super().__init__(config)

    def forward(self, x, k):
        return k

    def get_forward_flops(self, B, T) -> int:
        return 0

    def init_weights(self):
        pass


class DelegatingKeyAttention(AttentionStrategy):
    def __init__(
        self, config: GPTConfig, delegating_key_strategy: DelegatingKeyStrategy
    ):
        super().__init__(config)
        # Standard Projections
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        self.delegating_key_strategy = delegating_key_strategy

    def init_weights(self):
        s = uniform_normal_std_equalizer(self.config.n_embd)
        torch.nn.init.uniform_(self.c_q.weight, -s, s) # weights use Uniform to avoid outliers
        torch.nn.init.uniform_(self.c_k.weight, -s, s)
        torch.nn.init.uniform_(self.c_v.weight, -s, s)
        self.delegating_key_strategy.init_weights()

    def forward(self, x, cos_sin):
        B, T, _ = x.size()

        # 1. Project
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2. RoPE
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 3. Norm
        q, k = norm(q), norm(k)

        # 4. Cumsum gate
        k = self.delegating_key_strategy.forward(x, k)

        # 4. Transpose to (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def get_forward_flops(self, B, T):
        # 1. Q @ K.T -> (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        # Cost: 2 * B * H * T^2 * D
        qk_flops = 2 * B * self.n_head * (T**2) * self.head_dim

        # 2. Attn @ V -> (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        # Cost: 2 * B * H * T^2 * D
        # Note: We use n_head here because the result is expanded to n_head
        # even if V uses n_kv_head (due to broadcasting/GQA).
        av_flops = 2 * B * self.n_head * (T**2) * self.head_dim

        return qk_flops + av_flops
