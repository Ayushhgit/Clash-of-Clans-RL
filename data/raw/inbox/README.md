# Drop screenshots here

Any png / jpg / webp / bmp. Phone screenshots, desktop grabs, frames exported
from a recording -- all fine, mixed sizes are fine.

Then:

    python -m scripts.ingest --src data/raw/inbox --name manual_01

That copies them into `data/raw/screenshots/manual_01/` with a `meta.jsonl`,
which is the format the annotation tool and the detector trainer read.
Re-running after adding more files only ingests the new ones.

Useful flags:

    --state BASE_PREVIEW    tag every image with a screen state (Dataset B)
    --max-width 1920        downscale wider images (0 = keep native)
