"""Boundary-aware CRNN for sound event detection.

Two model families share one convolutional-recurrent backbone:

* `FrameCRNN` — the paper's frame-wise baseline: one presence logit per class.
* `BoundaryAwareCRNN` — two logits per class (event start / event end), a RED
  layer that converts them into presence/onset/offset probabilities, and an
  optional EPN that predicts onset/offset durations.

`forward` returns a dict so the training loop, evaluator and inference code can
consume RED/EPN outputs without knowing which ablation produced them.
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SEDConfig
from src.modules.epn import EventProposalNetwork
from src.modules.red import RED


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool(2, 2)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        """Build one convolutional block.

        Args:
            in_ch: Input feature maps.
            out_ch: Output feature maps.
            dropout: Spatial dropout probability after the activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv, batch-norm, ReLU, dropout and 2x2 max-pooling.

        Args:
            x: Feature map, `(B, in_ch, T, F)`.

        Returns:
            Pooled feature map, `(B, out_ch, T // 2, F // 2)`.
        """
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.drop(x)
        return F.max_pool2d(x, kernel_size=2, stride=2)


class CRNNBackbone(nn.Module):
    """4 conv blocks (time/frequency pooling) followed by a bidirectional GRU.

    Input `(B, 1, T, F)` -> output `(B, T, 2 * rnn_hidden)`. The recurrent output
    is interpolated back to the input frame rate so frame labels align.
    """

    def __init__(self, cfg: SEDConfig) -> None:
        """Build the convolutional and recurrent stack.

        Args:
            cfg: Architecture config with channel widths and GRU size.
        """
        super().__init__()
        self.cfg = cfg
        blocks: List[ConvBlock] = []
        in_ch = 1
        for i, out_ch in enumerate(cfg.cnn_channels):
            blocks.append(ConvBlock(in_ch, out_ch, dropout=0.0 if i == 0 else 0.1))
            in_ch = out_ch
        self.convs = nn.Sequential(*blocks)

        # Frequency is halved by every 2x2 pool; the GRU sees the flattened
        # channel x frequency map at each pooled time step.
        reduced_freq = max(cfg.n_mels // (2 ** len(cfg.cnn_channels)), 1)
        self.rnn_input_dim = in_ch * reduced_freq
        self.rnn = nn.GRU(
            self.rnn_input_dim,
            cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if cfg.rnn_layers == 1 else cfg.dropout,
        )
        self.out_dim = 2 * cfg.rnn_hidden
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a log-mel batch at the original frame rate.

        Args:
            x: Log-mel features, `(B, 1, T, F)`.

        Returns:
            Frame features `(B, T, 2 * rnn_hidden)`.

        Raises:
            ValueError: If the input is not 4-D or is too short for four
                pooling stages.
        """
        # x: (B, 1, T, F)
        if x.dim() != 4:
            raise ValueError(f"expected (B, 1, T, F), got {tuple(x.shape)}")
        target_frames = x.shape[2]
        h = self.convs(x)  # (B, C, T', F')
        b, c, t, f = h.shape
        if t < 1 or f < 1:
            raise ValueError(f"input too short for the conv stack: {tuple(x.shape)}")
        h = h.permute(0, 2, 1, 3).reshape(b, t, c * f)
        if c * f != self.rnn_input_dim:
            raise ValueError(
                "input n_mels does not match the configured frontend: "
                f"pooled features have {c * f} channels, expected {self.rnn_input_dim}"
            )
        h, _ = self.rnn(h)
        h = self.drop(h)
        h = h.transpose(1, 2)  # (B, 2H, T')
        h = F.interpolate(h, size=target_frames, mode="linear", align_corners=False)
        return h.transpose(1, 2)  # (B, T, 2H)


class FrameCRNN(nn.Module):
    """Frame-wise baseline: log-mel -> presence logits `(B, C, T)`."""

    def __init__(self, cfg: SEDConfig) -> None:
        """Build the frame-wise baseline.

        Args:
            cfg: Architecture config with `num_classes` set.

        Raises:
            ValueError: If `cfg.num_classes <= 0`.
        """
        super().__init__()
        if cfg.num_classes <= 0:
            raise ValueError("SEDConfig.num_classes must be > 0 before building")
        self.cfg = cfg
        self.backbone = CRNNBackbone(cfg)
        self.head = nn.Linear(self.backbone.out_dim, cfg.num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate frame features.

        Args:
            x: Log-mel features, `(B, 1, T, F)`.

        Returns:
            Frame features `(B, T, 2 * rnn_hidden)`.
        """
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Predict per-class presence logits.

        Args:
            x: Log-mel features, `(B, 1, T, F)`.
            lengths: Optional valid lengths `(B,)`; accepted for interface
                parity with the RED model.

        Returns:
            A dict with `pres_logits` `(B, C, T)` and `feature` `(B, T, H)`.
        """
        feat = self.backbone(x)
        logits = self.head(feat).transpose(1, 2)  # (B, C, T)
        return {"pres_logits": logits, "feature": feat}


class BoundaryAwareCRNN(nn.Module):
    """CRNN + RED (+ optional EPN) from arXiv:2601.04178."""

    def __init__(self, cfg: SEDConfig) -> None:
        """Build the RED/EPN model.

        Args:
            cfg: Architecture config with `num_classes` set and optional
                `use_red` / `use_epn` ablations.

        Raises:
            ValueError: If `cfg.num_classes <= 0`.
        """
        super().__init__()
        if cfg.num_classes <= 0:
            raise ValueError("SEDConfig.num_classes must be > 0 before building")
        self.cfg = cfg
        self.backbone = CRNNBackbone(cfg)
        self.head = nn.Linear(self.backbone.out_dim, 2 * cfg.num_classes)
        self.red = RED()
        self.epn = EventProposalNetwork(cfg) if cfg.use_epn else None

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate frame features.

        Args:
            x: Log-mel features, `(B, 1, T, F)`.

        Returns:
            Frame features `(B, T, 2 * rnn_hidden)`.
        """
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Predict boundaries, presence and (optionally) event durations.

        Args:
            x: Log-mel features, `(B, 1, T, F)`.
            lengths: Optional valid lengths `(B,)`; padded frames are masked.

        Returns:
            A dict with `start_logits` / `end_logits` / `presence` / `onset` /
            `offset` / `feature`, plus `don` / `doff` when EPN is enabled.
        """
        feat = self.backbone(x)
        b, t, _ = feat.shape
        c = self.cfg.num_classes
        se = self.head(feat).reshape(b, t, c, 2).permute(0, 2, 3, 1)
        start_logits, end_logits = se[:, :, 0, :], se[:, :, 1, :]  # (B, C, T)

        mask = None
        if lengths is not None:
            mask = torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        red = self.red(start_logits, end_logits, mask=mask)
        out = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "presence": red["presence"],
            "onset": red["onset"],
            "offset": red["offset"],
            "feature": feat,
        }
        if self.epn is not None:
            don, doff = self.epn(red["presence"], red["onset"], red["offset"])
            out["don"] = don
            out["doff"] = doff
        return out


def build_model(cfg: SEDConfig) -> nn.Module:
    """Build the model family selected by `cfg.use_red`.

    Args:
        cfg: Architecture config with `num_classes` set.

    Returns:
        `BoundaryAwareCRNN` when `use_red` is true, else `FrameCRNN`.
    """
    return BoundaryAwareCRNN(cfg) if cfg.use_red else FrameCRNN(cfg)


def presence_probability(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Read presence probabilities from either model family.

    Args:
        outputs: A model output dict.

    Returns:
        Presence probabilities, `(B, C, T)`.

    Raises:
        KeyError: If the dict has neither `presence` nor `pres_logits`.
    """
    if "presence" in outputs:
        return outputs["presence"]
    if "pres_logits" in outputs:
        return torch.sigmoid(outputs["pres_logits"])
    raise KeyError("model output has neither 'presence' nor 'pres_logits'")
