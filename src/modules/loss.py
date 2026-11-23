"""Boundary-aware SED losses (arXiv:2601.04178, Sec. II-B/II-C).

    L_total = L_pres + lambda_ool * (L_on + L_off) + lambda_iou * L_iou

* `L_pres`  — frame-wise binary cross-entropy on RED presence probabilities
              (or on the frame-wise baseline's sigmoid logits).
* `L_on/off` — focal loss (alpha=2) on the impulse-like onset/offset labels.
* `L_iou`   — duration-weighted IoU loss between the ground-truth interval
              `[t - d_on, t + d_off]` and the EPN proposal, on active frames.

The IoU loss is computed in closed form because both intervals contain `t`:
`IoU = (min(don, don_hat) + min(doff, doff_hat)) / (max(...) + max(...))`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SEDConfig
from src.modules.model import presence_probability

EPS = 1e-7


def focal_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary focal loss, averaged over valid elements.

    `probs`/`targets` are `(..., T)`; `mask` broadcasts over the leading dims.
    """
    probs = probs.clamp(EPS, 1.0 - EPS)
    pos = (1.0 - probs).pow(alpha) * probs.log()
    neg = probs.pow(alpha) * torch.log1p(-probs)
    loss = -(targets * pos + (1.0 - targets) * neg)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def iou_loss(
    don: torch.Tensor,
    doff: torch.Tensor,
    don_gt: torch.Tensor,
    doff_gt: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Duration-weighted IoU loss on active frames (Eq. 4).

    All inputs are `(B, C, T)`. `active` is the ground-truth presence mask.
    """
    inter = torch.minimum(don, don_gt) + torch.minimum(doff, doff_gt)
    union = torch.maximum(don, don_gt) + torch.maximum(doff, doff_gt)
    iou = inter / union.clamp(min=EPS)
    weight = don_gt + doff_gt
    per_frame = (1.0 - iou) * weight
    denom = active.sum().clamp(min=1.0)
    return (per_frame * active).sum() / denom


class SEDLoss(nn.Module):
    """Combined presence + onset/offset + IoU objective.

    `use_ool` / `use_epn` follow the architecture switches; the ablation
    `BL -> +RED -> +OOL -> +EPN` is exactly:
        use_red=False
        use_red=True,  use_ool=False, use_epn=False
        use_red=True,  use_ool=True,  use_epn=False
        use_red=True,  use_ool=True,  use_epn=True
    """

    def __init__(
        self,
        cfg: SEDConfig,
        lambda_ool: float = 100.0,
        lambda_iou: float = 2.0,
        focal_alpha: float = 2.0,
        use_ool: Optional[bool] = None,
        use_epn: Optional[bool] = None,
    ) -> None:
        """Configure the combined objective.

        Args:
            cfg: Architecture config; supplies the `use_red` / `use_epn`
                defaults for the ablation switches.
            lambda_ool: Weight on the summed onset + offset focal losses.
            lambda_iou: Weight on the duration-weighted IoU loss.
            focal_alpha: Focusing parameter for the boundary focal loss.
            use_ool: Override for the onset/offset loss switch.
            use_epn: Override for the IoU loss switch.

        Raises:
            ValueError: If the IoU loss is enabled without a RED model.
        """
        super().__init__()
        self.cfg = cfg
        self.lambda_ool = float(lambda_ool)
        self.lambda_iou = float(lambda_iou)
        self.focal_alpha = float(focal_alpha)
        self.use_ool = cfg.use_red if use_ool is None else bool(use_ool)
        self.use_epn = cfg.use_epn if use_epn is None else bool(use_epn)
        if self.use_epn and not cfg.use_red:
            raise ValueError("IoU loss needs EPN outputs, which need RED")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute every loss component and their weighted sum.

        Args:
            outputs: Model output dict.
            batch: Target dict with `pres`, `on`, `off`, `don`, `doff`, each
                `(B, C, T)`.
            mask: Optional boolean `(B, T)` valid-frame mask.

        Returns:
            A dict with scalar tensors `loss`, `pres`, `ool` and `iou`.
        """
        pres = presence_probability(outputs).clamp(EPS, 1.0 - EPS)
        y_pres = batch["pres"].float()
        if mask is None:
            mask_frames = torch.ones_like(y_pres)
        else:
            mask_frames = mask.unsqueeze(1).to(y_pres.dtype)

        if mask is None:
            loss_pres = F.binary_cross_entropy(pres, y_pres)
        else:
            bce = F.binary_cross_entropy(pres, y_pres, reduction="none")
            loss_pres = (bce * mask_frames).sum() / mask_frames.sum().clamp(min=1.0)

        zero = pres.new_zeros(())
        loss_ool = zero
        if self.use_ool:
            if "onset" not in outputs or "offset" not in outputs:
                raise KeyError("use_ool=True requires a RED model")
            focal_mask = None if mask is None else mask.unsqueeze(1)
            loss_on = focal_loss(
                outputs["onset"], batch["on"].float(), self.focal_alpha, focal_mask
            )
            loss_off = focal_loss(
                outputs["offset"], batch["off"].float(), self.focal_alpha, focal_mask
            )
            loss_ool = loss_on + loss_off

        loss_iou = zero
        if self.use_epn:
            if "don" not in outputs or "doff" not in outputs:
                raise KeyError("EPN loss requires EPN outputs")
            loss_iou = iou_loss(
                outputs["don"],
                outputs["doff"],
                batch["don"].float(),
                batch["doff"].float(),
                y_pres * mask_frames,
            )

        total = loss_pres + self.lambda_ool * loss_ool + self.lambda_iou * loss_iou
        return {
            "loss": total,
            "pres": loss_pres,
            "ool": loss_ool,
            "iou": loss_iou,
        }
