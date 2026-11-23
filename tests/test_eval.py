import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import SEDConfig
from src.data import SEDDataset, build_targets, collate, write_meta
from src.modules.events import Event
from src.modules.model import BoundaryAwareCRNN
from src.pipelines.eval import eval_per_epoch, evaluate, format_report


def make_cfg(**overrides) -> SEDConfig:
    base = dict(
        num_classes=2,
        num_frames=100,
        n_mels=16,
        hop_length=400,
        cnn_channels=(4, 8),
        rnn_hidden=8,
        rnn_layers=1,
        epn_hidden=4,
        epn_layers=1,
        dropout=0.0,
    )
    base.update(overrides)
    return SEDConfig(**base)


def make_dataset(root, cfg, n=8):
    keys = ("pres", "on", "off", "don", "doff")
    targets = {key: np.zeros((n, cfg.num_classes, cfg.num_frames)) for key in keys}
    for i in range(n):
        built = build_targets([Event(i % 2, 0.5, 1.5)], cfg)
        for key in keys:
            targets[key][i] = built[key]
    np.save(
        root / "train_mel.npy",
        np.zeros((n, cfg.num_frames, cfg.n_mels), dtype=np.float16),
    )
    np.savez(
        root / "train_targets.npz",
        **{k: v.astype(np.float16) for k, v in targets.items()},
    )
    write_meta(root, cfg, ["a", "b"], {"train": n}, 0.0, 1.0, "synthetic")
    return SEDDataset(root, "train")


def test_eval_returns_finite_metrics(tmp_path):
    cfg = make_cfg()
    ds = make_dataset(tmp_path, cfg)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    model = BoundaryAwareCRNN(cfg).eval()
    result = eval_per_epoch(
        model,
        loader,
        torch.device("cpu"),
        cfg,
        settings={"method": "epn", "threshold": 0.5, "k": 5},
    )
    for key in ("psds1", "psds2", "collar_f1", "frame_f1", "frame_accuracy"):
        assert np.isfinite(result[key])
    assert len(result["predictions"]) == 8
    assert len(result["ground_truth"]) == 8


def test_evaluate_persists_json(tmp_path):
    cfg = make_cfg()
    ds = make_dataset(tmp_path, cfg)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    model = BoundaryAwareCRNN(cfg).eval()
    out = tmp_path / "metrics.json"
    result = evaluate(
        model,
        loader,
        torch.device("cpu"),
        cfg,
        settings={"method": "median", "threshold": 0.5, "filter_len": 3},
        save_json_path=out,
    )
    data = json.loads(out.read_text())
    assert "psds1" in data
    assert "predictions" not in data
    assert np.isfinite(result["collar_f1"])


def test_format_report_mentions_metrics(tmp_path):
    cfg = make_cfg()
    ds = make_dataset(tmp_path, cfg)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    model = BoundaryAwareCRNN(cfg).eval()
    result = eval_per_epoch(model, loader, torch.device("cpu"), cfg)
    report = format_report(result)
    assert "PSDS1" in report
    assert "collar-F1" in report
