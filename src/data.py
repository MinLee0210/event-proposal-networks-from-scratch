"""URBAN-SED data plumbing: strong annotations, log-mel features, frame targets.

Raw layout (after extracting the archive under `data/`):

    data/URBAN-SED_v2.0.0/
      audio/{train,validate,test}/soundscape_<split>_<...>.wav
      annotations/{train,validate,test}/soundscape_<split>_<...>.txt

Each annotation line is `start<TAB>end<TAB>label` in seconds. The preparation
script (`scripts/data/prepare_urbansed.py`) resamples to 16 kHz, computes a
64-band log-mel spectrogram, rasterises events into four frame-level targets,
and writes:

    data/processed/urbansed/
      train_mel.npy       (N, 500, 64) float16
      train_targets.npz   pres/on/off/don/doff (N, C, 500) float16
      ...
      meta.json           classes, frontend, split files, normalisation stats

`SEDDataset` memory-maps the features and loads the small target arrays once.
"""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import SEDConfig
from src.modules.events import Event, events_to_frames, merge_events

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torchaudio

URBAN_SED_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


def parse_annotation(path: str | Path, class_to_idx: Dict[str, int]) -> List[Event]:
    """Read a URBAN-SED simplified annotation (TSV: start, end, label)."""
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            onset, offset, label = float(parts[0]), float(parts[1]), parts[2]
            if label not in class_to_idx:
                raise ValueError(f"unknown class {label!r} in {path}")
            if offset > onset:
                events.append(Event(class_to_idx[label], onset, offset, 1.0))
    return events


def discover_split(
    root: str | Path,
    split: str,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """List `{"name", "audio", "events"}` for one URBAN-SED split."""
    root = Path(root)
    class_to_idx = class_to_idx or {c: i for i, c in enumerate(URBAN_SED_CLASSES)}
    ann_dir = root / "annotations" / split
    audio_dir = root / "audio" / split
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {ann_dir}")

    rows = []
    for ann_path in sorted(ann_dir.glob("*.txt")):
        # macOS archives contain AppleDouble metadata files named "._*.txt".
        if ann_path.name.startswith("._"):
            continue
        audio_path = audio_dir / f"{ann_path.stem}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"audio missing for {ann_path.name}: {audio_path}")
        rows.append(
            {
                "name": ann_path.stem,
                "audio": str(audio_path),
                "events": parse_annotation(ann_path, class_to_idx),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# frontend
# ---------------------------------------------------------------------------


def make_mel_transform(cfg: SEDConfig) -> "torchaudio.transforms.MelSpectrogram":
    """Build the torchaudio log-mel transform matching `SEDConfig`.

    Args:
        cfg: Frontend config with sample rate, FFT, hop and mel-band fields.

    Returns:
        A configured `MelSpectrogram` transform.
    """
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        n_mels=cfg.n_mels,
        power=2.0,
        center=True,
        normalized=False,
    )


def waveform_to_logmel(
    waveform: torch.Tensor,
    transform: "torchaudio.transforms.MelSpectrogram",
    num_frames: int,
) -> np.ndarray:
    """Convert a mono waveform into a fixed-length log-mel spectrogram.

    Args:
        waveform: Mono waveform tensor, `(samples,)` or `(1, samples)`.
        transform: Configured mel transform from `make_mel_transform`.
        num_frames: Number of output frames; shorter inputs are zero-padded.

    Returns:
        `(num_frames, n_mels)` float32 log-mel in dB.
    """
    if waveform.dim() == 2:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
    mel = transform(waveform.unsqueeze(0))  # (1, n_mels, T)
    mel = torch.log10(mel.clamp(min=1e-10)) * 10.0
    mel = mel.squeeze(0).transpose(0, 1)  # (T, n_mels)
    if mel.shape[0] < num_frames:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, num_frames - mel.shape[0]))
    return mel[:num_frames].contiguous().numpy().astype(np.float32)


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load a mono waveform at `sample_rate` (torchaudio, librosa fallback)."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).float()
    except Exception:  # noqa: BLE001 - fall back to librosa for odd codecs
        import librosa

        y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        return torch.from_numpy(y).float()


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def build_targets(events: Sequence[Event], cfg: SEDConfig) -> Dict[str, np.ndarray]:
    """Rasterise events into presence/onset/offset and duration targets.

    All arrays are `(C, T)` float32. Durations are seconds; active frames carry
    `t - onset` and `offset - t`, zero elsewhere.
    """
    c, t = cfg.num_classes, cfg.num_frames
    merged = merge_events(events)
    pres, on, off = events_to_frames(merged, c, t, cfg.frame_rate)
    pres_a = np.asarray(pres, dtype=np.float32)
    on_a = np.asarray(on, dtype=np.float32)
    off_a = np.asarray(off, dtype=np.float32)
    don = np.zeros((c, t), dtype=np.float32)
    doff = np.zeros((c, t), dtype=np.float32)

    for ev in merged:
        if not 0 <= ev.class_id < c:
            continue
        start = int(round(ev.onset * cfg.frame_rate))
        end = int(round(ev.offset * cfg.frame_rate))
        start = min(max(start, 0), t - 1)
        end = min(max(end, start + 1), t)
        frames = np.arange(start, end)
        don[ev.class_id, start:end] = (frames - start) / cfg.frame_rate
        doff[ev.class_id, start:end] = (end - frames) / cfg.frame_rate

    return {"pres": pres_a, "on": on_a, "off": off_a, "don": don, "doff": doff}


def targets_to_events(
    pres: np.ndarray, frame_rate: float, threshold: float = 0.5
) -> List[Event]:
    """Ground-truth events from a `(C, T)` presence array."""
    from src.modules.events import frames_to_events

    return frames_to_events(pres.tolist(), frame_rate, threshold=threshold)


# ---------------------------------------------------------------------------
# processed dataset
# ---------------------------------------------------------------------------


def processed_paths(data_root: str | Path, split: str) -> Tuple[Path, Path]:
    """Return the feature and target paths for a processed split.

    Args:
        data_root: Processed dataset directory.
        split: Split name.

    Returns:
        `(mel_path, targets_path)`.
    """
    root = Path(data_root)
    return root / f"{split}_mel.npy", root / f"{split}_targets.npz"


def events_path(data_root: str | Path, split: str) -> Path:
    """Return the raw-annotation event path for a processed split.

    Args:
        data_root: Processed dataset directory.
        split: Split name.

    Returns:
        Path to `events_<split>.json`.
    """
    return Path(data_root) / f"events_{split}.json"


def write_events(data_root: str | Path, split: str, rows: Sequence[dict]) -> None:
    """Persist raw strong annotations next to the processed features.

    The frame targets merge same-class overlaps (one presence region per class),
    but evaluation must count the original annotated events. This JSON is the
    evaluation ground truth; training still uses the rasterised targets.

    Args:
        data_root: Processed dataset directory.
        split: Split name.
        rows: `discover_split` rows with an `events` list.
    """
    payload = [
        [[int(ev.class_id), float(ev.onset), float(ev.offset)] for ev in row["events"]]
        for row in rows
    ]
    path = events_path(data_root, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_events(data_root: str | Path, split: str) -> List[List[Event]]:
    """Load raw annotated events for a processed split.

    Args:
        data_root: Processed dataset directory.
        split: Split name.

    Returns:
        One event list per clip. Returns an empty list when the sidecar file is
        absent (older processed datasets), so callers can fall back to frames.

    Raises:
        ValueError: If the JSON has an unexpected shape.
    """
    path = events_path(data_root, split)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of clips in {path}")
    return [[Event(int(c), float(s), float(e)) for c, s, e in clip] for clip in raw]


def load_meta(data_root: str | Path) -> dict:
    path = Path(data_root) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"processed metadata not found: {path}. "
            "Run scripts/data/prepare_urbansed.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SEDDataset(Dataset):
    """Precomputed log-mel + frame targets for one split."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        meta: Optional[dict] = None,
        mmap: bool = True,
    ) -> None:
        """Open a precomputed split.

        Args:
            data_root: Directory containing `<split>_mel.npy` and
                `<split>_targets.npz`.
            split: Split name (`train`, `validate`, `test`).
            meta: Parsed `meta.json`; loaded from `data_root` when omitted.
            mmap: Memory-map the feature array instead of loading it fully.

        Raises:
            FileNotFoundError: If the feature/target files are missing.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.meta = meta or load_meta(data_root)
        mel_path, target_path = processed_paths(data_root, split)
        if not mel_path.exists() or not target_path.exists():
            raise FileNotFoundError(
                f"missing processed split {split!r}: {mel_path} / {target_path}"
            )
        self.mel = np.load(mel_path, mmap_mode="r" if mmap else None)
        with np.load(target_path) as target_file:
            self.targets = {
                key: np.asarray(target_file[key], dtype=np.float32)
                for key in ("pres", "on", "off", "don", "doff")
            }
        self.mean = float(self.meta.get("mel_mean", 0.0))
        self.std = float(self.meta.get("mel_std", 1.0)) or 1.0
        self.events = load_events(data_root, split)

    def __len__(self) -> int:
        """Return the number of clips in the split.

        Returns:
            Clip count `N`.
        """
        return int(self.mel.shape[0])

    @property
    def num_classes(self) -> int:
        """Return the number of event classes.

        Returns:
            `C` from the target array.
        """
        return int(self.targets["pres"].shape[1])

    @property
    def num_frames(self) -> int:
        """Return the number of frames per clip.

        Returns:
            `T` from the target array.
        """
        return int(self.targets["pres"].shape[2])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Load and normalise one clip plus its four frame targets.

        Args:
            index: Row index into the precomputed arrays.

        Returns:
            A dict with `mel` `(1, T, F)` and `pres`/`on`/`off`/`don`/`doff`
            `(C, T)`, plus the integer `index`.
        """
        mel = np.asarray(self.mel[index], dtype=np.float32)
        mel = (mel - self.mean) / self.std
        return {
            "mel": torch.from_numpy(mel).unsqueeze(0),  # (1, T, F)
            "pres": torch.from_numpy(self.targets["pres"][index]),
            "on": torch.from_numpy(self.targets["on"][index]),
            "off": torch.from_numpy(self.targets["off"][index]),
            "don": torch.from_numpy(self.targets["don"][index]),
            "doff": torch.from_numpy(self.targets["doff"][index]),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def ground_truth_events(self, index: int, threshold: float = 0.5) -> List[Event]:
        """Return annotated events, falling back to the frame targets.

        Args:
            index: Clip index.
            threshold: Presence threshold used only when raw annotations are
                unavailable.

        Returns:
            Ground-truth events in seconds. Raw annotations keep same-class
            overlaps that the training targets intentionally merge.
        """
        if self.events:
            return self.events[index]
        return targets_to_events(
            self.targets["pres"][index], self.meta["frame_rate"], threshold
        )


def collate(batch: Sequence[dict]) -> dict:
    """Stack a list of dataset dicts into batched tensors.

    Args:
        batch: Non-empty list of `SEDDataset.__getitem__` outputs.

    Returns:
        A dict mapping each key to a stacked tensor.
    """
    keys = batch[0].keys()
    return {key: torch.stack([row[key] for row in batch]) for key in keys}


def write_meta(
    data_root: str | Path,
    cfg: SEDConfig,
    classes: Sequence[str],
    split_sizes: Dict[str, int],
    mel_mean: float,
    mel_std: float,
    source_root: str,
    files: Optional[Dict[str, Sequence[str]]] = None,
) -> None:
    """Write the processed-dataset `meta.json`.

    Args:
        data_root: Output directory.
        cfg: Frontend config used to build the features.
        classes: Ordered class names.
        split_sizes: Mapping split name to clip count.
        mel_mean: Global log-mel mean used for normalisation.
        mel_std: Global log-mel standard deviation.
        source_root: Path of the extracted URBAN-SED archive.
        files: Optional mapping split name to ordered source filenames.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "URBAN-SED v2.0.0",
        "source_root": source_root,
        "classes": list(classes),
        "num_classes": len(classes),
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "win_length": cfg.win_length,
        "hop_length": cfg.hop_length,
        "n_mels": cfg.n_mels,
        "f_min": cfg.f_min,
        "f_max": cfg.f_max,
        "num_frames": cfg.num_frames,
        "frame_rate": cfg.frame_rate,
        "clip_duration": cfg.num_frames / cfg.frame_rate,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "split_sizes": split_sizes,
        "files": {split: list(names) for split, names in (files or {}).items()},
    }
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# misc utilities
# ---------------------------------------------------------------------------


def read_csv_rows(path: str | Path) -> List[dict]:
    """Small helper for analysis notebooks (no pandas dependency)."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
