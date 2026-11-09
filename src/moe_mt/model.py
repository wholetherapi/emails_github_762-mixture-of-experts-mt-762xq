"""MoE encoder-decoder translator."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import TopKRouter


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class SparseMoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, shared_experts: int):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_model * 4) for _ in range(num_experts)])
        self.shared = nn.ModuleList(
            [ExpertFFN(d_model, d_model * 4) for _ in range(shared_experts)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, d_model = x.shape
        flat = x.reshape(-1, d_model)
        top_probs, top_indices, aux_loss = self.router(flat)

        out = torch.zeros_like(flat)
        for k in range(self.router.top_k):
            weight = top_probs[:, k].unsqueeze(-1)
            index = top_indices[:, k]
            for expert_id, expert in enumerate(self.experts):
                mask = index == expert_id
                if mask.any():
                    out[mask] += weight[mask] * expert(flat[mask])

        for expert in self.shared:
            out += expert(flat)

        out = self.norm(out.reshape(bsz, seq_len, d_model) + x)
        return out, aux_loss


class SparseMoETranslator(nn.Module):
    def __init__(self, d_model: int = 512, num_layers: int = 6,
                 num_experts: int = 16, top_k: int = 2, shared_experts: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(32000, d_model)
        self.layers = nn.ModuleList([
            SparseMoELayer(d_model, num_experts, top_k, shared_experts)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, 32000)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        x = self.embedding(input_ids)
        aux_losses = []
        for layer in self.layers:
            x, aux_loss = layer(x)
            aux_losses.append(aux_loss)
        logits = self.lm_head(x)
        return {
            "logits": logits,
            "aux_loss": torch.stack(aux_losses).mean() if aux_losses else None,
        }
