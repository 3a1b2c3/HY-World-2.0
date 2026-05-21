"""Download HY-World 2.0 model weights from HuggingFace.

Pulls subsets of `tencent/HY-World-2.0` (the unified repo) into the local
checkpoint tree, plus the external Qwen3-VL-8B-Instruct used by worldgen's
trajectory planning stage.

Skips files already on disk (snapshot_download is resume-aware). Safe to re-run.

HF subfolders inside tencent/HY-World-2.0:
  HY-WorldMirror-2.0/  - WorldMirror-2 reconstruction (~1.2B params)
  HY-Pano-2.0/         - HY-Pano-2 full image-to-pano model (~80B params, BIG)
                         + pytorch_lora_weights.safetensors (~425M LoRA variant)

External:
  Qwen/Qwen3-VL-8B-Instruct - VLM served via vLLM for traj_generate.py (worldgen)

Env-var overrides:
  HYWORLD_CKPT_DIR  destination root  (default: <repo>/checkpoint)
  HF_TOKEN          for gated repos   (snapshot_download picks it up automatically)
"""

import argparse
import os
import sys
from pathlib import Path

import fnmatch

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR = Path(os.environ.get("HYWORLD_CKPT_DIR", REPO_ROOT / "checkpoint"))

HYWORLD_REPO = "tencent/HY-World-2.0"
QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"


def _download(repo_id: str, dest: Path, label: str, allow: list[str] | None = None) -> bool:
    sentinel = dest / "config.json"
    if sentinel.exists() and not allow:
        print(f"[skip] {label}: {dest} already populated")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    label_full = f"{label} ({repo_id}" + (f" {allow}" if allow else "") + ")"
    print(f"[get ] {label_full} -> {dest}")
    # Avoid huggingface_hub.snapshot_download — it always spawns a tqdm
    # ThreadPoolExecutor which deadlocks on Windows + Python 3.12 with
    # "RuntimeError: cannot join thread before it is started". Sequential
    # per-file hf_hub_download has no executor and runs reliably.
    api = HfApi()
    info = api.repo_info(repo_id)
    files = [s.rfilename for s in info.siblings]
    if allow:
        files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in allow)]
    print(f"  pulling {len(files)} file(s) sequentially ...")
    for i, fname in enumerate(files, 1):
        hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(dest))
        print(f"  [{i}/{len(files)}] {fname}")
    print(f"[done] {label}: {dest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mirror", action="store_true",
                        help="WorldMirror-2 reconstruction weights (~1.2B). Needed for worldrecon + worldgen stage 5.")
    parser.add_argument("--pano", action="store_true",
                        help="HY-Pano-2 full model (~80B params, ~150 GB+). Use --pano-lora instead if you only need Qwen-backed pano.")
    parser.add_argument("--pano-lora", action="store_true",
                        help="HY-Pano-2 LoRA-only variant (~425M, ~850 MB). Pairs with Qwen-Image-Edit backend.")
    parser.add_argument("--qwen", action="store_true",
                        help="Qwen3-VL-8B-Instruct (~16 GB) for vLLM trajectory planning in worldgen.")
    parser.add_argument("--all", action="store_true",
                        help="Download everything except --pano (use --pano explicitly because it's huge).")
    args = parser.parse_args()

    # No flag => treat as --all (mirror + pano-lora + qwen, but NOT the 80 B
    # --pano which the user must opt into explicitly).
    if not (args.mirror or args.pano or args.pano_lora or args.qwen or args.all):
        print("[info] no flag given; defaulting to --all (mirror + pano-lora + qwen). "
              "Add --pano explicitly for the 80 B full pano model.")
        args.all = True

    if args.all:
        args.mirror = True
        args.pano_lora = True
        args.qwen = True

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        print("FATAL: HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) env var not set.", file=sys.stderr)
        print("  Get a token at https://huggingface.co/settings/tokens then:", file=sys.stderr)
        print("    set HF_TOKEN=hf_xxx", file=sys.stderr)
        return 2

    print(f"CKPT_DIR = {CKPT_DIR}")
    print()

    failures: list[str] = []

    if args.mirror:
        try:
            _download(
                HYWORLD_REPO,
                CKPT_DIR / "HY-WorldMirror-2.0",
                "HY-WorldMirror-2.0",
                allow=["HY-WorldMirror-2.0/*"],
            )
        except Exception as e:
            print(f"[FAIL] HY-WorldMirror-2.0: {e}", file=sys.stderr)
            failures.append("HY-WorldMirror-2.0")

    if args.pano:
        try:
            _download(
                HYWORLD_REPO,
                CKPT_DIR / "HY-Pano-2.0",
                "HY-Pano-2.0 (full)",
                allow=["HY-Pano-2.0/*"],
            )
        except Exception as e:
            print(f"[FAIL] HY-Pano-2.0 (full): {e}", file=sys.stderr)
            failures.append("HY-Pano-2.0 (full)")

    if args.pano_lora:
        try:
            _download(
                HYWORLD_REPO,
                CKPT_DIR / "HY-Pano-2.0",
                "HY-Pano-2.0 (LoRA only)",
                allow=["HY-Pano-2.0/pytorch_lora_weights.safetensors"],
            )
        except Exception as e:
            print(f"[FAIL] HY-Pano-2.0 (LoRA): {e}", file=sys.stderr)
            failures.append("HY-Pano-2.0 (LoRA)")

    if args.qwen:
        try:
            _download(QWEN_REPO, CKPT_DIR / "Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct")
        except Exception as e:
            print(f"[FAIL] Qwen3-VL-8B-Instruct: {e}", file=sys.stderr)
            failures.append("Qwen3-VL-8B-Instruct")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("All requested weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
