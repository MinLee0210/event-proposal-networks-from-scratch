"""Event Proposal Networks (EPN).

EPN (arXiv:2601.04178, Sec. II-C) predicts two non-negative durations at every
frame: time since the event onset (`d_on`) and time to the event offset
(`d_off`). The interval `[t - d_on, t + d_off]` is a region proposal that is
trained with the duration-weighted IoU loss. Two variants from the paper:

* `per_class`: one bidirectional GRU per class, input `(B, T, 3)`.
* `single`: one bidirectional GRU over all classes, input `(B, T, 3C)`,
  output `(B, T, 2C)`; used for many-class datasets.

The GRUs consume the RED probabilities `[presence, onset, offset]`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SEDConfig


class _DurationHead(nn.Module):
    """Bidirectional GRU followed by a linear head with Softplus output."""

    def __init__(self, input_dim: int, hidden: int, layers: int, out_dim: int) -> None:
        """Build one duration head.

        Args:
            input_dim: Number of input channels per frame.
            hidden: GRU hidden size per direction.
            layers: Number of stacked GRU layers.
            out_dim: Number of non-negative outputs per frame.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if layers == 1 else 0.1,
        )
        self.head = nn.Linear(2 * hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict non-negative durations for a sequence.

        Args:
            x: Frame features, `(N, T, input_dim)`.

        Returns:
            `(N, T, out_dim)` values after Softplus.
        """
        # x: (N, T, input_dim) -> (N, T, out_dim), Softplus keeps durations >= 0.
        out, _ = self.gru(x)
        return F.softplus(self.head(out))


class EventProposalNetwork(nn.Module):
    """Per-class or single-GRU duration predictor.

    Inputs are the RED outputs `(B, C, T)`; outputs `don`, `doff` are
    `(B, C, T)` in seconds.
    """

    def __init__(self, cfg: SEDConfig) -> None:
        """Build the EPN variant requested by `cfg.epn_variant`.

        Args:
            cfg: Architecture config; `epn_variant` selects one GRU per class
                or one GRU over all class channels.

        Raises:
            ValueError: If `epn_variant` is not one of `per_class` / `single`.
        """
        super().__init__()
        self.cfg = cfg
        if cfg.epn_variant == "per_class":
            self.heads = nn.ModuleList(
                [
                    _DurationHead(3, cfg.epn_hidden, cfg.epn_layers, 2)
                    for _ in range(cfg.num_classes)
                ]
            )
        elif cfg.epn_variant == "single":
            self.head = _DurationHead(
                3 * cfg.num_classes,
                cfg.epn_hidden,
                cfg.epn_layers,
                2 * cfg.num_classes,
            )
        else:  # guarded by SEDConfig, kept for clarity
            raise ValueError(f"unknown epn_variant: {cfg.epn_variant!r}")

    def forward(
        self,
        presence: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict time-since-onset and time-to-offset at every frame.

        Args:
            presence: RED presence probabilities, `(B, C, T)`.
            onset: RED onset probabilities, `(B, C, T)`.
            offset: RED offset probabilities, `(B, C, T)`.

        Returns:
            `(don, doff)`, each `(B, C, T)` in seconds and non-negative.
        """
        # (B, C, T) -> (B, T, C)
        feats = torch.stack([presence, onset, offset], dim=-1).permute(0, 2, 1, 3)
        # feats: (B, T, C, 3)

        if self.cfg.epn_variant == "per_class":
            outs = []
            for c, head in enumerate(self.heads):
                outs.append(head(feats[:, :, c, :]))
            durations = torch.stack(outs, dim=2)  # (B, T, C, 2)
        else:
            flat = feats.reshape(feats.shape[0], feats.shape[1], -1)
            durations = self.head(flat).reshape(
                feats.shape[0], feats.shape[1], self.cfg.num_classes, 2
            )

        durations = durations.permute(0, 2, 1, 3)  # (B, C, T, 2)
        return durations[..., 0], durations[..., 1]
