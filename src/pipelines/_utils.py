"""Presentation helpers for the SED training entrypoint.

Kept separate from `src.pipelines.train` so the training loop stays readable.
Nothing here affects training — it only prints configs, a data sample and a
forward-shape check. `rich` is optional; plain aligned text is the fallback.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import SEDConfig
from src.pipelines.config import TrainingConfig

try:  # optional pretty-printing dependency
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_RICH = False

WIDTH = 78


def banner(title: str) -> None:
    """Print a plain section banner.

    Args:
        title: Section title.
    """
    print(f"\n{title}\n{'-' * WIDTH}")


def indent(text: str, spaces: int = 2) -> str:
    """Indent every line of a block of text.

    Args:
        text: Multi-line text.
        spaces: Number of leading spaces.

    Returns:
        The indented text.
    """
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def announce_training(
    *,
    model: torch.nn.Module,
    model_cfg: SEDConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    train_dataset: Dataset,
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    """Print the run configuration, a data sample and the model layout.

    Args:
        model: Model that will be trained.
        model_cfg: Architecture config.
        train_cfg: Training config.
        train_loader: Training loader (used for one sample batch).
        train_dataset: Training dataset.
        device: Torch device.
        run_name: Run identifier.
        ckpt_dir: Checkpoint directory.
        result_dir: Result directory.
    """
    common = dict(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
    )
    if _HAS_RICH:
        _announce_rich(**common)
    else:
        _announce_plain(**common)


def _announce_plain(
    *,
    model: torch.nn.Module,
    model_cfg: SEDConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    train_dataset: Dataset,
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    """Plain-text fallback for `announce_training`."""
    print("\n" + "=" * WIDTH)
    print(f"RUN  {run_name}")
    print("=" * WIDTH)
    print(f"device:      {device}   seed: {train_cfg.seed}")
    print(f"checkpoints: {ckpt_dir}")
    print(f"results:     {result_dir}")

    banner("model config (SEDConfig)")
    print(indent(json.dumps(asdict(model_cfg), indent=2, default=str)))
    banner("training config (TrainingConfig)")
    print(indent(json.dumps(asdict(train_cfg), indent=2, default=str)))

    batch = next(iter(train_loader))
    mel = batch["mel"]
    banner(
        f"data: train sample ({len(train_dataset)} clips, {model_cfg.num_classes} classes)"
    )
    print(f"batch: mel {tuple(mel.shape)} {mel.dtype}")
    print(
        f"mel stats: min={mel.min():+.3f}  max={mel.max():+.3f}  "
        f"mean={mel.mean():+.3f}  std={mel.std():.3f}"
    )
    active = batch["pres"].mean().item()
    print(
        f"targets: presence active fraction={active * 100:.2f}%  "
        f"onsets/batch={int(batch['on'].sum())}  offsets/batch={int(batch['off'].sum())}"
    )

    banner(f"model: {type(model).__name__}")
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        print(f"  {name:<38s} {str(tuple(param.shape)):>16s}  {param.numel():>10,d}")
    print(f"  {'TOTAL':<38s} {'':>16s}  {total:>10,d}")

    banner("forward check (untrained)")
    print(_forward_check(model, mel, device))
    print("\n" + "=" * WIDTH + "\n")


def _forward_check(
    model: torch.nn.Module, x: torch.Tensor, device: torch.device
) -> str:
    """Run one no-grad forward pass and describe the output shapes.

    Args:
        model: Model under inspection.
        x: A real input batch.
        device: Torch device.

    Returns:
        A human-readable shape summary; never raises.
    """
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            outputs = model(x.to(device))
        model.train(was_training)
        parts = [f"mel {tuple(x.shape)}"]
        for key in ("pres_logits", "presence", "onset", "offset", "don", "doff"):
            if key in outputs:
                parts.append(f"{key} {tuple(outputs[key].shape)}")
        return "  ".join(parts)
    except Exception as e:  # noqa: BLE001 - a shape peek must never block training
        return f"forward check skipped: {e}"


def _announce_rich(
    *,
    model: torch.nn.Module,
    model_cfg: SEDConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    train_dataset: Dataset,
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    """Rich rendering of the training preamble."""
    console = Console(highlight=False)
    console.print()
    console.print(Rule(f"[bold]RUN  {run_name}[/bold]"))

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan", justify="right")
    info.add_column()
    info.add_row("device", str(device))
    info.add_row("seed", str(train_cfg.seed))
    info.add_row("checkpoints", str(ckpt_dir))
    info.add_row("results", str(result_dir))
    console.print(info)

    console.print(
        Panel(
            JSON(json.dumps(asdict(model_cfg), default=str)),
            title="model config (SEDConfig)",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            JSON(json.dumps(asdict(train_cfg), default=str)),
            title="training config (TrainingConfig)",
            border_style="blue",
        )
    )

    batch = next(iter(train_loader))
    mel = batch["mel"]
    table = Table(
        title=f"data: train sample ({len(train_dataset)} clips)", border_style="green"
    )
    table.add_column("property", style="bold")
    table.add_column("value", justify="right")
    table.add_row("mel shape", str(tuple(mel.shape)))
    table.add_row("mel range", f"{mel.min():+.3f} .. {mel.max():+.3f}")
    table.add_row("mel mean/std", f"{mel.mean():+.3f} / {mel.std():.3f}")
    table.add_row("presence active", f"{batch['pres'].mean().item() * 100:.2f}%")
    table.add_row("onsets / batch", str(int(batch["on"].sum())))
    table.add_row("offsets / batch", str(int(batch["off"].sum())))
    console.print(table)

    model_table = Table(title=f"model: {type(model).__name__}", border_style="magenta")
    model_table.add_column("layer")
    model_table.add_column("shape", justify="right")
    model_table.add_column("params", justify="right")
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        model_table.add_row(name, str(tuple(param.shape)), f"{param.numel():,d}")
    model_table.add_section()
    model_table.add_row("TOTAL", "", f"{total:,d}", style="bold")
    console.print(model_table)
    console.print(
        Panel(
            _forward_check(model, mel, device),
            title="forward check (untrained)",
            border_style="yellow",
        )
    )
    console.print()
