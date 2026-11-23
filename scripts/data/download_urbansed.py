"""Download and extract the URBAN-SED v2.0.0 archive.

    uv run python scripts/data/download_urbansed.py

The 6.6 GB archive is written to `data/raw/` and extracted to
`data/URBAN-SED_v2.0.0/` (the layout used by `prepare_urbansed.py`). The script
is idempotent: existing files are reused unless `--force` is passed.
"""

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

REPO_ID = "LakoreAI/URBAN-SED"
FILENAME = "URBAN-SED_v2.0.0.tar.gz"
EXPECTED_MD5 = "a2d24a2148ece7c021fcc079ee87c2dc"
EXPECTED_BYTES = 6_641_129_513


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 digest of a file.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes.

    Returns:
        Hex digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Download, verify and extract the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    archive = args.raw_dir / FILENAME
    if args.force or not archive.exists():
        print(f"downloading {REPO_ID}/{FILENAME} -> {archive}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=str(args.raw_dir),
        )
    else:
        print(f"reusing {archive}")

    size = archive.stat().st_size
    print(f"archive size: {size:,} bytes")
    if size != EXPECTED_BYTES:
        print(f"warning: expected {EXPECTED_BYTES:,} bytes", file=sys.stderr)
    if not args.skip_verify:
        print("verifying md5 ...")
        digest = md5sum(archive)
        if digest != EXPECTED_MD5:
            raise SystemExit(f"md5 mismatch: {digest} != {EXPECTED_MD5}")
        print("md5 ok")

    if args.skip_extract:
        return
    marker = args.extract_dir / "URBAN-SED_v2.0.0" / "audio" / "train"
    if marker.exists() and not args.force:
        print(f"already extracted: {args.extract_dir / 'URBAN-SED_v2.0.0'}")
        return
    print(f"extracting -> {args.extract_dir}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(args.extract_dir, filter="data")
    print("done")


if __name__ == "__main__":
    main()
