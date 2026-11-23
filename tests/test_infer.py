import json

import numpy as np
import torch

from src.config import SEDConfig
from src.data import SEDDataset, build_targets, write_events, write_meta
from src.modules.events import Event
from src.modules.model import BoundaryAwareCRNN
from src.pipelines.infer import config_from_checkpoint, infer, load_model
from src.utils.io_utils import save_checkpoint


def make_dataset(root, cfg, n=2):
    keys = ("pres", "on", "off", "don", "doff")
    targets = {key: np.zeros((n, cfg.num_classes, cfg.num_frames)) for key in keys}
    rows = []
    for i in range(n):
        events = [Event(0, 0.2, 0.8)]
        built = build_targets(events, cfg)
        for key in keys:
            targets[key][i] = built[key]
        rows.append({"events": events})
    np.save(
        root / "test_mel.npy",
        np.zeros((n, cfg.num_frames, cfg.n_mels), dtype=np.float16),
    )
    np.savez(
        root / "test_targets.npz",
        **{k: v.astype(np.float16) for k, v in targets.items()},
    )
    write_events(root, "test", rows)
    write_meta(root, cfg, ["a", "b"], {"test": n}, 0.0, 1.0, "synthetic")
    return SEDDataset(root, "test")


def test_checkpoint_roundtrip_and_infer(tmp_path):
    cfg = SEDConfig(
        num_classes=2,
        num_frames=50,
        n_mels=8,
        hop_length=400,
        cnn_channels=(4, 8),
        rnn_hidden=8,
        rnn_layers=1,
        epn_hidden=4,
        epn_layers=1,
        dropout=0.0,
    )
    ds = make_dataset(tmp_path, cfg)
    model = BoundaryAwareCRNN(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "best.pt"
    save_checkpoint(
        model,
        opt,
        0,
        ckpt,
        extra={
            "model_cfg": {
                "num_classes": 2,
                "num_frames": 50,
                "n_mels": 8,
                "hop_length": 400,
                "cnn_channels": (4, 8),
                "rnn_hidden": 8,
                "rnn_layers": 1,
                "epn_hidden": 4,
                "epn_layers": 1,
                "dropout": 0.0,
            }
        },
    )
    loaded, loaded_cfg = load_model(ckpt, torch.device("cpu"))
    assert loaded_cfg.num_classes == 2
    assert config_from_checkpoint(torch.load(ckpt, map_location="cpu")).num_frames == 50

    out = tmp_path / "events.json"
    result = infer(
        ckpt,
        split="test",
        index=0,
        data_root=str(tmp_path),
        method="epn",
        k=3,
        out_path=out,
    )
    assert "events" in result
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["source"] == "test[0]"
    assert len(ds.events) == 2
