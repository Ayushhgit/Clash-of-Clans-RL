"""Ingest hand-supplied screenshots into a capture session (Phase 2 substitute).

The rest of the pipeline (annotation tool, dataset loaders, detector training)
reads a session directory: numbered PNGs plus a `meta.jsonl`. When frames come
from a live capture, `scripts/record_session.py` writes that. When they are
dropped in by hand -- screenshots taken on a phone, pasted from chat, exported
from a recording -- this script builds the same structure.

    # drop images anywhere, then:
    python -m scripts.ingest --src data/raw/inbox --name manual_01

    # optionally tag them all with a screen state
    python -m scripts.ingest --src data/raw/inbox --state BASE_PREVIEW

Accepts png/jpg/jpeg/webp/bmp. Originals are left alone; converted copies go
to the session. Duplicate images (same bytes) are skipped, so re-running after
adding a few more files is safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

from ai.control.capture import SCREEN_STATES

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def iter_images(src: pathlib.Path):
    if src.is_file():
        yield src
        return
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in EXTS and p.parent.name != "__ingested__":
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/inbox",
                    help="file or folder of screenshots to ingest")
    ap.add_argument("--root", default="data/raw/screenshots")
    ap.add_argument("--name", default=None, help="session name (default: manual_<date>)")
    ap.add_argument("--state", default="UNKNOWN", choices=SCREEN_STATES,
                    help="screen state to stamp on every ingested frame")
    ap.add_argument("--max-width", type=int, default=1920,
                    help="downscale wider images; 0 keeps native size")
    args = ap.parse_args()

    from PIL import Image

    src = pathlib.Path(args.src)
    if not src.exists():
        src.mkdir(parents=True, exist_ok=True)
        raise SystemExit(f"created {src} -- put screenshots there and rerun")

    name = args.name or time.strftime("manual_%Y%m%d")
    out = pathlib.Path(args.root) / name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"

    seen = set()
    start = 0
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                seen.add(d.get("sha1", ""))
                start = max(start, d["frame_id"] + 1)

    n_new, n_dup = 0, 0
    for p in iter_images(src):
        raw = p.read_bytes()
        sha = hashlib.sha1(raw).hexdigest()
        if sha in seen:
            n_dup += 1
            continue

        img = Image.open(p).convert("RGB")
        if args.max_width and img.width > args.max_width:
            h = round(img.height * args.max_width / img.width)
            img = img.resize((args.max_width, h), Image.LANCZOS)

        frame_id = start + n_new
        rel = f"{frame_id:09d}.png"
        img.save(out / rel)
        with meta_path.open("a") as f:
            f.write(json.dumps({
                "frame_id": frame_id,
                "timestamp": float(frame_id),
                "wall_time": p.stat().st_mtime,
                "screen_state": args.state,
                "path": rel,
                "sha1": sha,
                "source": str(p),
                "width": img.width,
                "height": img.height,
            }) + "\n")
        seen.add(sha)
        n_new += 1

    print(f"ingested {n_new} new frames ({n_dup} duplicates skipped) -> {out}")
    if n_new:
        print("next:")
        print(f"  python -m annotation.app {out}          # draw boxes")
        print("  python -m scripts.train_detector --data data/raw/screenshots")


if __name__ == "__main__":
    main()
