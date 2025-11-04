#!/usr/bin/env python3
"""
Downloads your Release assets into data/raw/ using direct URLs.
Update the ASSETS list if you add more files to the release.
"""
import os
import pathlib
import requests
from tqdm import tqdm

ASSETS = [
    # Your current release assets (GM-data-11-05)
    "https://github.com/mitties2020/Personal-Assistant/releases/download/GM-data-11-05/Harrison.s.Principles.of.Internal.Medicine.18e.pdf",
    "https://github.com/mitties2020/Personal-Assistant/releases/download/GM-data-11-05/Mechanisms.of.Clinical.Signs.pdf",
    "https://github.com/mitties2020/Personal-Assistant/releases/download/GM-data-11-05/Murtagh.s.General.Practice.5th.Edition.pdf",
    "https://github.com/mitties2020/Personal-Assistant/releases/download/GM-data-11-05/Talley._.O.Connor.s.Clinical.Examination.7e.pdf",
]

DEST = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DEST.mkdir(parents=True, exist_ok=True)

for url in ASSETS:
    name = url.split("/")[-1]
    out = DEST / name
    if out.exists() and out.stat().st_size > 1024:
        print(f"[skip] {name} already exists")
        continue
    print(f"[download] {name}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
print("Done. Files saved to data/raw/")
