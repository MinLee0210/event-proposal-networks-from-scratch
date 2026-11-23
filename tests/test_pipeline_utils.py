from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import SEDConfig
from src.data import SEDDataset, collate, write_meta
from src.modules.model import BoundaryAwareCRNN
from src.pipelines._utils import announce_training, banner, indent
from src.pipelines.config import TrainingConfig


def _dataset(root: Path) -> SEDDataset:
    n, t, f, c = 4, 32, 8, 2
    np.save(root / "train_mel.npy", np.zeros((n, t, f), dtype=np.float16))
    np.savez(
        root / "train_targets.npz",
        **{
            key: np.zeros((n, c, t), dtype=np.float16)
            for key in ("pres", "on", "off", "don", "doff")
        },
    )
    cfg = SEDConfig(
        num_classes=c,
        num_frames=t,
        n_mels=f,
        cnn_channels=(4, 8),
        rnn_hidden=8,
        rnn_layers=1,
        epn_hidden=4,
        epn_layers=1,
        dropout=0.0,
        hop_length=400,
    )
    write_meta(root, cfg, ["a", "b"], {"train": n}, 0.0, 1.0, "synthetic")
    return SEDDataset(root, "train")


def test_indent_and_banner(capsys):
    banner("section")
    print(indent("a\nb", 2))
    out = capsys.readouterr().out
    assert "section" in out
    assert "  a\n  b" in out


def test_announce_training_prints_config_and_shapes(tmp_path, capsys):
    ds = _dataset(tmp_path)
    cfg = SEDConfig(
        num_classes=2,
        num_frames=32,
        n_mels=8,
        cnn_channels=(4, 8),
        rnn_hidden=8,
        rnn_layers=1,
        epn_hidden=4,
        epn_layers=1,
        dropout=0.0,
        hop_length=400,
    )
    model = BoundaryAwareCRNN(cfg)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    announce_training(
        model=model,
        model_cfg=cfg,
        train_cfg=TrainingConfig(seed=1),
        train_loader=loader,
        train_dataset=ds,
        device=torch.device("cpu"),
        run_name="run-x",
        ckpt_dir=Path("ckpt"),
        result_dir=Path("res"),
    )
    out = capsys.readouterr().out
    assert "RUN  run-x" in out
    assert "SEDConfig" in out
    assert "forward check" in out
    assert "presence" in out
