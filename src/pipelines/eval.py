"""SED evaluation: PSDS1, collar F1, frame metrics and per-class event F1.

`eval_per_epoch` runs one pass over a loader, converts frame probabilities into
events with the configured extractor (EPN Algorithm 1 or median filtering), and
returns a plain metrics dict. `evaluate` is the persistence wrapper.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import SEDConfig
from src.modules.events import Event, frames_to_events
from src.modules.loss import SEDLoss
from src.modules.metrics import collar_f1, frame_metrics, psds1, psds2
from src.modules.model import presence_probability
from src.modules.postprocess import events_from_model_output
from src.utils.io_utils import save_json


@torch.no_grad()
def eval_per_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: SEDConfig,
    settings: Optional[dict] = None,
    criterion: Optional[SEDLoss] = None,
    gt_events_by_index: Optional[Sequence[Sequence[Event]]] = None,
) -> Dict[str, object]:
    """Evaluate one split and return event and frame metrics.

    Args:
        model: Frame or boundary-aware CRNN.
        loader: DataLoader over an `SEDDataset`.
        device: Torch device.
        cfg: Architecture config (frame rate, clip length, class count).
        settings: Post-processing settings; defaults to EPN with `k=15`.
        criterion: Optional loss module; the returned `loss` is NaN without it.
        gt_events_by_index: Raw annotated events per dataset row. When omitted,
            ground truth is recovered from the merged presence targets.

    Returns:
        A dict with `loss`, `psds1`, `collar_f1`, `frame_f1`,
        `frame_precision`, `frame_recall`, `per_class_collar`, `psds_curve`,
        `predictions` and `ground_truth` event lists.
    """
    model.eval()
    settings = dict(settings or {})
    method = settings.get("method", "epn")
    presence_all, target_all = [], []
    losses, pred_events, gt_events = [], [], []

    for batch in loader:
        mel = batch["mel"].float().to(device)
        outputs = model(mel)
        if criterion is not None:
            targets_dev = {
                key: batch[key].float().to(device)
                for key in ("pres", "on", "off", "don", "doff")
            }
            losses.append(
                float(criterion(outputs, targets_dev, mask=None)["loss"].item())
            )
        presence = presence_probability(outputs).cpu()
        targets = batch["pres"].float().cpu()
        presence_all.append(presence.numpy())
        target_all.append(targets.numpy())

        # Fall back to median filtering for models without EPN outputs.
        batch_method = method
        if batch_method == "epn" and "don" not in outputs:
            batch_method = "median"
        pred_events.extend(
            events_from_model_output(
                outputs,
                cfg,
                method=batch_method,
                threshold=float(settings.get("threshold", 0.5)),
                filter_len=int(settings.get("filter_len", 7)),
                k=int(settings.get("k", 15)),
                m=settings.get("m"),
                min_duration=float(settings.get("min_duration", 0.0)),
            )
        )
        if gt_events_by_index is not None:
            gt_events.extend(
                [gt_events_by_index[int(i)] for i in batch["index"].tolist()]
            )
        else:
            gt_events.extend(
                [
                    frames_to_events(
                        targets[i].numpy().tolist(),
                        cfg.frame_rate,
                        threshold=0.5,
                    )
                    for i in range(targets.shape[0])
                ]
            )

    if not presence_all:
        return {
            "loss": float("nan"),
            "psds1": float("nan"),
            "psds2": float("nan"),
            "collar_f1": float("nan"),
            "frame_f1": float("nan"),
            "frame_precision": float("nan"),
            "frame_recall": float("nan"),
            "per_class_collar": {},
            "predictions": [],
            "ground_truth": [],
        }

    presence_np = np.concatenate(presence_all, axis=0)
    target_np = np.concatenate(target_all, axis=0)
    frame = frame_metrics(presence_np, target_np, threshold=0.5)

    duration = len(gt_events) * (cfg.num_frames / cfg.frame_rate)
    psds_result = psds1(pred_events, gt_events, dataset_duration=duration)
    psds2_result = psds2(pred_events, gt_events, dataset_duration=duration)
    collar = collar_f1(
        pred_events,
        gt_events,
        threshold=float(settings.get("threshold", 0.5)),
        collar=0.2,
    )

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "psds1": psds_result["psds1"],
        "psds_curve": psds_result["curve"],
        "psds2": psds2_result["psds2"],
        "psds2_curve": psds2_result["curve"],
        "collar_f1": collar["f1"],
        "collar_precision": collar["precision"],
        "collar_recall": collar["recall"],
        "collar_threshold": collar["threshold"],
        "per_class_collar": collar["per_class"],
        "frame_f1": frame["f1"],
        "frame_precision": frame["precision"],
        "frame_recall": frame["recall"],
        "frame_accuracy": frame["accuracy"],
        "predictions": pred_events,
        "ground_truth": gt_events,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: SEDConfig,
    settings: Optional[dict] = None,
    criterion: Optional[SEDLoss] = None,
    save_json_path: Optional[Union[str, Path]] = None,
    gt_events_by_index: Optional[Sequence[Sequence[Event]]] = None,
) -> Dict[str, object]:
    """Evaluate a model and optionally persist the metrics dict.

    Args:
        model: Model to evaluate.
        loader: Evaluation loader.
        device: Torch device.
        cfg: Architecture config.
        settings: Post-processing settings.
        criterion: Optional loss module.
        save_json_path: When set, write the metrics dict here.
        gt_events_by_index: Raw annotated events per dataset row.

    Returns:
        The metrics dict from `eval_per_epoch`.
    """
    result = eval_per_epoch(
        model,
        loader,
        device,
        cfg,
        settings=settings,
        criterion=criterion,
        gt_events_by_index=gt_events_by_index,
    )
    if save_json_path is not None:
        serialisable = {
            k: v for k, v in result.items() if k not in ("predictions", "ground_truth")
        }
        save_json(serialisable, save_json_path)
    return result


def format_report(result: Dict[str, object]) -> str:
    """Render a compact one-block summary of an evaluation result.

    Args:
        result: Metrics dict from `eval_per_epoch`.

    Returns:
        A multi-line string suitable for stdout.
    """
    lines = [
        f"PSDS1={result.get('psds1', float('nan')) * 100:6.2f}  "
        f"PSDS2={result.get('psds2', float('nan')) * 100:6.2f}  "
        f"collar-F1={result.get('collar_f1', float('nan')) * 100:6.2f}  "
        f"frame-F1={result.get('frame_f1', float('nan')) * 100:6.2f}  "
        f"loss={result.get('loss', float('nan')):.4f}"
    ]
    per_class = result.get("per_class_collar", {})
    for cls, row in sorted(per_class.items()):
        lines.append(
            f"  class {int(cls):>3d}  F1={row['f1'] * 100:6.2f}  "
            f"P={row['precision'] * 100:6.2f}  R={row['recall'] * 100:6.2f}"
        )
    return "\n".join(lines)
