"""Train a boundary-aware CRNN on URBAN-SED.

The loop is deliberately small: one forward pass returns RED/EPN outputs, the
combined loss is `L_pres + lambda_ool * (L_on + L_off) + lambda_iou * L_iou`,
and periodic validation converts frame probabilities into events to compute
PSDS1 and collar F1.

Usage:
    uv run python -m src.pipelines.train --config configs/crnn_urbansed.yaml
    uv run python -m src.pipelines.train --config configs/crnn_smoke.yaml
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.callbacks import (
    BestCheckpoint,
    Callback,
    EarlyStopping,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.callbacks.wandb_callback import WandbCallback
from src.data import SEDDataset, collate, load_meta
from src.modules.augment import mixup_batch, spec_augment
from src.modules.loss import SEDLoss
from src.modules.model import build_model
from src.pipelines._utils import announce_training
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import eval_per_epoch, format_report
from src.utils.io_utils import load_env, save_checkpoint, save_json
from src.utils.model_utils import detect_device, get_run_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_loaders(
    train_cfg: TrainingConfig, device: torch.device
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], SEDDataset, dict]:
    """Build train/val/test loaders from the processed dataset.

    Args:
        train_cfg: Training configuration.
        device: Torch device (controls `pin_memory`).

    Returns:
        `(train_loader, val_loader, test_loader, train_dataset, meta)`.

    Raises:
        FileNotFoundError: If the processed dataset is missing.
    """
    meta = load_meta(train_cfg.data_root)
    train_ds = SEDDataset(train_cfg.data_root, train_cfg.train_split, meta=meta)
    val_ds = SEDDataset(train_cfg.data_root, train_cfg.val_split, meta=meta)
    test_ds = SEDDataset(train_cfg.data_root, train_cfg.test_split, meta=meta)

    pin_memory = train_cfg.pin_memory and device.type == "cuda"
    common = dict(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        persistent_workers=train_cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=len(train_ds) >= train_cfg.batch_size,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common) if len(val_ds) else None
    test_loader = DataLoader(test_ds, shuffle=False, **common) if len(test_ds) else None
    return train_loader, val_loader, test_loader, train_ds, meta


def build_callbacks(
    train_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    run_name: str,
    wandb_config: dict,
    total_steps: int,
) -> list[Callback]:
    """Assemble LR scheduling, early stopping, checkpointing and W&B callbacks.

    Args:
        train_cfg: Training configuration.
        optimizer: Optimizer whose LR is scheduled.
        run_name: Run name for logging.
        wandb_config: Serializable config passed to W&B.
        total_steps: Total optimizer steps (for the cosine schedule).

    Returns:
        A list of callbacks in execution order.
    """
    callbacks = []
    scheduler_cfg = dict(train_cfg.lr_scheduler or {})
    if scheduler_cfg.get("type") == "warmup_cosine":
        scheduler_cfg.setdefault("warmup_steps", train_cfg.warmup_steps)
        scheduler_cfg.setdefault("t_max", total_steps)
    lr_cb = build_lr_scheduler(optimizer, scheduler_cfg)
    if lr_cb is not None:
        callbacks.append(lr_cb)

    if train_cfg.early_stopping and train_cfg.early_stopping.get("enabled", True):
        es_kwargs = {
            k: v for k, v in train_cfg.early_stopping.items() if k != "enabled"
        }
        callbacks.append(EarlyStopping(**es_kwargs))

    if train_cfg.save_best:
        callbacks.append(
            BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
        )

    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "epn-sed"),
                run_name=run_name,
                config=wandb_config,
                entity=wb.get("entity"),
                monitor=wb.get("monitor", train_cfg.best_metric),
                mode=wb.get("mode", train_cfg.best_mode),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
            )
        )
    return callbacks


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    callbacks: list[Callback],
    ctx: TrainContext,
    pending_losses: list[float],
    step: int,
    epoch: int,
    log_every: int,
    grad_clip: float,
    pending_components: Optional[list[dict]] = None,
) -> tuple[int, float]:
    """Apply one accumulated update and notify step-level callbacks.

    Args:
        optimizer: Optimizer to step.
        scaler: AMP gradient scaler.
        callbacks: Active callbacks.
        ctx: Training context.
        pending_losses: Loss values accumulated since the last step.
        pending_components: Per-step loss component dicts for the same window.
        step: Current global step.
        epoch: Current epoch.
        log_every: Logging interval in steps.
        grad_clip: Gradient-norm clip; 0 disables.

    Returns:
        `(step, average_loss)` after the update.
    """
    if grad_clip and grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            grad_clip,
        )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    step += 1
    avg_loss = sum(pending_losses) / len(pending_losses)
    extra = {}
    if pending_components:
        keys = pending_components[0].keys()
        extra = {
            f"loss_{key}": sum(row[key] for row in pending_components)
            / len(pending_components)
            for key in keys
        }
    state = TrainerState(step=step, train_loss=avg_loss, epoch=epoch, extra=extra)
    for cb in callbacks:
        cb.on_step_end(ctx, state)
    if step % log_every == 0 or step == 1:
        print(f"  epoch {epoch:3d}  step {step:5d}  loss={avg_loss:.4f}")
    return step, avg_loss


def train_per_epoch(
    model: torch.nn.Module,
    criterion: SEDLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    accum_steps: int,
    device: torch.device,
    callbacks: list[Callback],
    ctx: TrainContext,
    step: int,
    epoch: int,
    log_every: int,
    grad_clip: float,
    train_cfg_augment: Optional[dict] = None,
) -> tuple[float, dict[str, float], int]:
    """Run one training epoch.

    Args:
        model: SED model.
        criterion: `SEDLoss` module.
        loader: Training loader.
        optimizer: AdamW optimizer.
        scaler: AMP gradient scaler.
        accum_steps: Gradient accumulation steps.
        device: Torch device.
        callbacks: Active callbacks.
        ctx: Training context.
        step: Starting global step.
        epoch: Current epoch.
        log_every: Logging interval in steps.
        grad_clip: Gradient-norm clip.

    Returns:
        `(average_loss, components, step)` where `components` averages
        `pres`/`ool`/`iou`.
    """
    model.train()
    epoch_losses, pending = [], []
    pending_components: list[dict] = []
    component_sums = {"pres": 0.0, "ool": 0.0, "iou": 0.0}
    component_count = 0
    optimizer.zero_grad()

    augment = train_cfg_augment or {}
    for batch in loader:
        if augment.get("mixup_prob", 0.0) > 0:
            batch = mixup_batch(batch, prob=float(augment["mixup_prob"]))
        mel = batch["mel"].float().to(device)
        if augment.get("time_masks", 0) or augment.get("freq_masks", 0):
            mel = spec_augment(
                mel,
                time_masks=int(augment.get("time_masks", 0)),
                time_width=int(augment.get("time_width", 50)),
                freq_masks=int(augment.get("freq_masks", 0)),
                freq_width=int(augment.get("freq_width", 8)),
            )
        targets = {
            key: batch[key].float().to(device)
            for key in ("pres", "on", "off", "don", "doff")
        }
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(mel)
        # Compute BCE/focal/IoU outside autocast: BCE is unsafe in fp16.
        losses = criterion(outputs, targets, mask=None)
        loss = losses["loss"]

        scaler.scale(loss / accum_steps).backward()
        pending.append(loss.item())
        pending_components.append(
            {key: float(losses[key].detach()) for key in component_sums}
        )
        component_count += 1
        for key in component_sums:
            component_sums[key] += float(losses[key].detach())

        if len(pending) >= accum_steps:
            step, avg_loss = _optimizer_step(
                optimizer,
                scaler,
                callbacks,
                ctx,
                pending,
                step,
                epoch,
                log_every,
                grad_clip,
                pending_components,
            )
            epoch_losses.append(avg_loss)
            pending = []
            pending_components = []

    if pending:
        step, avg_loss = _optimizer_step(
            optimizer,
            scaler,
            callbacks,
            ctx,
            pending,
            step,
            epoch,
            log_every,
            grad_clip,
            pending_components,
        )
        epoch_losses.append(avg_loss)

    components = {
        key: value / max(component_count, 1) for key, value in component_sums.items()
    }
    mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
    return mean_loss, components, step


def train(train_cfg: TrainingConfig) -> list[dict]:
    """Run the full training pipeline and return the per-epoch history.

    Args:
        train_cfg: Resolved training configuration.

    Returns:
        A list of per-epoch history records (also written to JSON).
    """
    load_env(REPO_ROOT / ".env")
    device = detect_device()
    print(f"device: {device}")

    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(train_cfg.seed)

    train_loader, val_loader, test_loader, train_ds, meta = build_loaders(
        train_cfg, device
    )
    model_cfg = train_cfg.model_config(train_ds.num_classes)
    if model_cfg.num_frames != train_ds.num_frames:
        raise ValueError(
            "processed features and model disagree on num_frames: "
            f"{train_ds.num_frames} vs {model_cfg.num_frames}"
        )
    model = build_model(model_cfg).to(device)
    loss_kwargs = dict(train_cfg.loss or {})
    loss_kwargs.setdefault("use_ool", model_cfg.use_red)
    loss_kwargs.setdefault("use_epn", model_cfg.use_epn)
    criterion = SEDLoss(model_cfg, **loss_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=(train_cfg.amp and device.type == "cuda"),
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * train_cfg.epochs

    run_name = train_cfg.run_name or get_run_name(
        "crnn_red_epn" if model_cfg.use_red else "crnn_baseline",
        Path(train_cfg.data_root).name,
        train_cfg.lr,
        train_cfg.batch_size,
    )
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name

    callbacks = build_callbacks(
        train_cfg,
        optimizer,
        run_name,
        {"model": asdict(model_cfg), "training": asdict(train_cfg)},
        total_steps,
    )
    early_stoppers = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]

    wandb_run_id = None
    for cb in callbacks:
        if isinstance(cb, WandbCallback) and cb.enabled and cb.run is not None:
            wandb_run_id = cb.run.id

    ctx = TrainContext(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict={"model": asdict(model_cfg), "training": asdict(train_cfg)},
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    if train_cfg.resume_from:
        raw = torch.load(train_cfg.resume_from, map_location=str(device))
        model.load_state_dict(raw["model"])
        if "optimizer" in raw:
            optimizer.load_state_dict(raw["optimizer"])
        step = int(raw.get("step", 0))
        start_epoch = int(raw.get("extra", {}).get("epoch", 0)) + 1
        print(f"resumed at step {step}, continuing from epoch {start_epoch}")
    else:
        step, start_epoch = 0, 1

    announce_training(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_ds,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
    )

    history = []
    stopped_early = False
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_loss, components, step = train_per_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            train_cfg.accum_steps,
            device,
            callbacks,
            ctx,
            step,
            epoch,
            train_cfg.log_every,
            train_cfg.grad_clip,
            train_cfg.augment,
        )
        record = {
            "epoch": epoch,
            "step": step,
            "loss": train_loss,
            **{f"loss_{k}": v for k, v in components.items()},
        }
        print(
            f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}  "
            f"pres={components['pres']:.4f}  ool={components['ool']:.4f}  "
            f"iou={components['iou']:.4f}"
        )

        should_validate = epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs
        if should_validate and val_loader is not None:
            settings = train_cfg.eval_settings()
            val_gt = getattr(val_loader.dataset, "events", None) or None
            val_result = eval_per_epoch(
                model,
                val_loader,
                device,
                model_cfg,
                settings=settings,
                criterion=criterion,
                gt_events_by_index=val_gt,
            )
            extra = {
                "psds1": val_result["psds1"],
                "psds2": val_result["psds2"],
                "collar_f1": val_result["collar_f1"],
                "frame_f1": val_result["frame_f1"],
                "metric_loss": val_result["loss"],
            }
            record.update(extra)
            record["val_loss"] = val_result["loss"]
            print(format_report(val_result))

            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=val_result["loss"],
                extra=extra,
            )
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            if any(es.should_stop for es in early_stoppers):
                stopped_early = True

        history.append(record)

        if epoch % train_cfg.ckpt_every == 0 or epoch == train_cfg.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                step,
                ckpt_path,
                extra={
                    "cfg": {"model": asdict(model_cfg), "training": asdict(train_cfg)},
                    "model_cfg": asdict(model_cfg),
                    "epoch": epoch,
                    "wandb_run_id": ctx.wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_path}")

        if stopped_early:
            break

    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")

    if test_loader is not None and val_loader is not None:
        # Evaluate the best checkpoint on the held-out test split.
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            raw = torch.load(best_path, map_location=str(device))
            model.load_state_dict(raw["model"])
        test_gt = getattr(test_loader.dataset, "events", None) or None
        test_result = eval_per_epoch(
            model,
            test_loader,
            device,
            model_cfg,
            settings=train_cfg.eval_settings(),
            criterion=criterion,
            gt_events_by_index=test_gt,
        )
        save_json(
            {
                k: v
                for k, v in test_result.items()
                if k not in ("predictions", "ground_truth")
            },
            result_dir / "test_metrics.json",
        )
        print("\ntest split\n" + format_report(test_result))

        # Log the test scalars while the W&B run is still open.
        scalar_keys = (
            "psds1",
            "psds2",
            "collar_f1",
            "collar_precision",
            "collar_recall",
            "frame_f1",
            "frame_precision",
            "frame_recall",
            "frame_accuracy",
            "loss",
        )
        test_scalars = {
            k: float(test_result[k])
            for k in scalar_keys
            if k in test_result and test_result[k] is not None
        }
        for cb in callbacks:
            if isinstance(cb, WandbCallback) and cb.enabled:
                cb.log_metrics(test_scalars, step=step, prefix="test")

    for cb in callbacks:
        cb.on_train_end(ctx)

    print(
        f"\n{'stopped early' if stopped_early else 'finished'} "
        f"at epoch {epoch} (step {step})"
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_training_config(args.config, **overrides)
    train(cfg)
