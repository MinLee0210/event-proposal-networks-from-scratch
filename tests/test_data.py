import numpy as np
import pytest
import torch

from src.config import SEDConfig
from src.data import (
    SEDDataset,
    build_targets,
    collate,
    load_events,
    load_meta,
    parse_annotation,
    write_events,
    write_meta,
)
from src.modules.events import Event, frames_to_events, merge_events


def make_cfg(**overrides) -> SEDConfig:
    base = dict(num_classes=3, num_frames=100, n_mels=16, hop_length=400)
    base.update(overrides)
    return SEDConfig(**base)


def test_build_targets_values():
    cfg = SEDConfig(num_classes=1, num_frames=500, hop_length=320)
    targets = build_targets([Event(0, 1.0, 3.0)], cfg)
    assert targets["pres"].shape == (1, 500)
    start = int(round(1.0 * cfg.frame_rate))
    end = int(round(3.0 * cfg.frame_rate))
    assert targets["pres"][0].sum() == end - start
    assert targets["on"][0, start] == 1.0
    assert targets["off"][0, end] == 1.0
    assert targets["don"][0, start] == 0.0
    assert abs(targets["doff"][0, start] - 2.0) < 1e-5
    assert targets["don"][0, start - 1] == 0.0


def test_merge_overlapping_events():
    merged = merge_events([Event(0, 0.0, 2.0), Event(0, 1.0, 3.0), Event(1, 1.0, 2.0)])
    assert len(merged) == 2
    assert merged[0].class_id == 0
    assert merged[0].offset == 3.0


def test_frames_to_events_roundtrip():
    cfg = SEDConfig(num_classes=1, num_frames=500, hop_length=320)
    targets = build_targets([Event(0, 1.0, 3.0)], cfg)
    events = frames_to_events(targets["pres"], cfg.frame_rate)
    assert len(events) == 1
    assert abs(events[0].onset - 1.0) < 1e-6
    assert abs(events[0].offset - 3.0) < 1e-6


def test_parse_annotation(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("1.0\t2.5\tdog_bark\n0.0\t1.0\tsiren\n")
    class_to_idx = {"dog_bark": 0, "siren": 1}
    events = parse_annotation(path, class_to_idx)
    assert len(events) == 2
    assert events[0].class_id == 0 and events[0].onset == 1.0


def test_parse_annotation_rejects_unknown_class(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("1.0\t2.0\tunknown\n")
    with pytest.raises(ValueError):
        parse_annotation(path, {"siren": 0})


def _write_processed(root, cfg):
    n = 4
    mel = np.arange(n * cfg.num_frames * cfg.n_mels, dtype=np.float32).reshape(
        n, cfg.num_frames, cfg.n_mels
    )
    np.save(root / "train_mel.npy", mel.astype(np.float16))
    keys = ("pres", "on", "off", "don", "doff")
    targets = {
        key: np.zeros((n, cfg.num_classes, cfg.num_frames), dtype=np.float16)
        for key in keys
    }
    np.savez(root / "train_targets.npz", **targets)
    write_meta(
        root,
        cfg,
        ["a", "b", "c"],
        {"train": n},
        1.0,
        2.0,
        "synthetic",
    )
    return mel


def test_sed_dataset_roundtrip(tmp_path):
    cfg = make_cfg()
    raw = _write_processed(tmp_path, cfg)
    meta = load_meta(tmp_path)
    ds = SEDDataset(tmp_path, "train", meta=meta)
    assert len(ds) == 4
    assert ds.num_classes == 3
    assert ds.num_frames == 100
    row = ds[0]
    expected = (raw[0] - 1.0) / 2.0
    assert torch.allclose(row["mel"].squeeze(0), torch.from_numpy(expected))
    assert row["pres"].shape == (3, 100)


def test_collate_stacks_dicts():
    batch = [
        {"mel": torch.zeros(1, 5, 2), "index": torch.tensor(0)},
        {"mel": torch.ones(1, 5, 2), "index": torch.tensor(1)},
    ]
    stacked = collate(batch)
    assert stacked["mel"].shape == (2, 1, 5, 2)
    assert stacked["index"].tolist() == [0, 1]


def test_missing_meta_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_meta(tmp_path)


def test_events_sidecar_roundtrip(tmp_path):
    cfg = make_cfg()
    _write_processed(tmp_path, cfg)
    rows = [
        {"events": [Event(0, 1.0, 2.0)]},
        {"events": [Event(1, 0.5, 1.0), Event(1, 1.5, 2.0)]},
        {"events": []},
        {"events": [Event(2, 2.0, 2.5)]},
    ]
    write_events(tmp_path, "train", rows)
    events = load_events(tmp_path, "train")
    assert len(events) == 4
    assert len(events[1]) == 2 and events[1][0].class_id == 1
    ds = SEDDataset(tmp_path, "train")
    assert ds.ground_truth_events(1) == events[1]
