"""Top-k sparse routing used by the mixture-of-experts translator."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Route every token to its top-k experts with a load-balancing loss."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 jitter_noise: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        logits = self.gate(hidden_states)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        # Fraction of tokens routed to each expert, used for the auxiliary loss.
        one_hot = F.one_hot(top_indices, num_classes=self.num_experts).float()
        router_prob_per_expert = one_hot.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(
            router_prob_per_expert * probs.mean(dim=(0, 1)))

        return top_probs, top_indices, aux_loss

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, top_k={self.top_k}"
