"""
Download HY-World-2.0 model weights from HuggingFace.

Currently available:
  - HY-WorldMirror-2.0  (WorldMirror 2.0 — multi-view/video -> 3D reconstruction, ~1.2B params)

Coming soon (not yet released):
  - HY-Pano-2           (Panorama generation)
  - WorldStereo-2       (World expansion)

Usage:
    python download_models.py
    python download_models.py --model worldmirror
    python download_models.py --local_dir ./checkpoints
"""

import argparse
import os
from huggingface_hub import snapshot_download

REPO_ID = "tencent/HY-World-2.0"

MODELS = {
    "worldmirror": {
        "subfolder": "HY-WorldMirror-2.0",
        "description": "WorldMirror 2.0 — multi-view/video -> 3D reconstruction (~1.2B params)",
    },
}


def download(model_key: str, local_dir: str | None = None):
    info = MODELS[model_key]
    subfolder = info["subfolder"]
    print(f"Downloading {subfolder} ({info['description']}) from {REPO_ID} ...")

    kwargs = dict(
        repo_id=REPO_ID,
        allow_patterns=[f"{subfolder}/*"],
    )
    if local_dir:
        kwargs["local_dir"] = local_dir

    path = snapshot_download(**kwargs)
    resolved = os.path.join(path, subfolder) if not local_dir else os.path.join(local_dir, subfolder)
    print(f"Saved to: {resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Download HY-World-2.0 model weights")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to download (default: all available)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Optional local directory to save models into (default: HuggingFace cache)",
    )
    args = parser.parse_args()

    keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    for key in keys:
        download(key, args.local_dir)
    print("Done.")


if __name__ == "__main__":
    main()
