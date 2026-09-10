#!/usr/bin/env python3
"""Download Hillstrom MineThatData and write a small committed sample + synthetic RCT."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_uplift.data.load import (  # noqa: E402
    HILLSTROM_URL,
    SAMPLE_DIR,
    RAW_DIR,
    generate_synthetic_rct,
)

# Public mirrors if the original host is flaky
MIRRORS = [
    HILLSTROM_URL,
    # GitHub raw mirrors used by various uplift tutorials
    "https://raw.githubusercontent.com/duketwelve/uplift_modeling/master/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    "https://gist.githubusercontent.com/diogojc/71d1d86285acb311eba3/raw/hillstrom.csv",
]


def download_hillstrom(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in MIRRORS:
        try:
            print(f"Trying {url} ...")
            urllib.request.urlretrieve(url, dest)
            # sanity
            text = dest.read_text(errors="ignore")[:200]
            if "recency" in text or "history" in text:
                print(f"Saved {dest} ({dest.stat().st_size} bytes)")
                return dest
            last_err = RuntimeError(f"Unexpected content from {url}")
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"Could not download Hillstrom. Last error: {last_err}")


def write_sample(full_csv: Path, sample_path: Path, n: int = 2000, seed: int = 42):
    import pandas as pd

    df = pd.read_csv(full_csv)
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)
    print(f"Wrote sample {sample_path} ({len(sample)} rows)")


def write_synthetic(path: Path, n: int = 5000, seed: int = 42):
    df = generate_synthetic_rct(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote synthetic RCT {path} ({len(df)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true", help="Only write synthetic + use existing")
    p.add_argument("--sample-n", type=int, default=2000)
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    full = RAW_DIR / "hillstrom.csv"
    if not args.skip_download:
        try:
            download_hillstrom(full)
            write_sample(full, SAMPLE_DIR / "hillstrom_sample.csv", n=args.sample_n)
        except Exception as e:
            print(f"WARNING: Hillstrom download failed ({e}). Falling back to synthetic-as-sample.")
            syn = generate_synthetic_rct(n=args.sample_n, seed=42)
            syn.to_csv(SAMPLE_DIR / "hillstrom_sample.csv", index=False)
            print("Wrote synthetic stand-in to data/sample/hillstrom_sample.csv")
    elif full.exists():
        write_sample(full, SAMPLE_DIR / "hillstrom_sample.csv", n=args.sample_n)

    write_synthetic(RAW_DIR / "synthetic_rct.csv", n=20000, seed=42)
    write_synthetic(SAMPLE_DIR / "synthetic_sample.csv", n=3000, seed=42)
    print("Done.")


if __name__ == "__main__":
    main()
