"""Short-query encoder used by query-aware latent selection."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    """Pool a short question with a frozen or trainable Transformer backbone.

    The module receives only question tokens.  Passing a context argument is
    intentionally impossible, preventing accidental source-context rereads in
    the online selector path.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have shape [B, Q]")
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token-level hidden states ``[B, Q, D]`` for slot construction.

        Slot policies need per-token query representations instead of the
        pooled vector.  The backbone follows the same freeze/no-grad discipline
        as :meth:`forward`; only the pooling differs.
        """
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have shape [B, Q]")
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask
