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
  hanshanxue/WorldStereo    - video-diffusion DiT used by worldgen stage 3
                              (video_gen.py). Lives in the HF cache, not
                              CKPT_DIR, because that's where from_pretrained
                              looks at runtime. Default variant subfolder is
                              ``worldstereo-memory-dmd`` (3-step distilled).

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
WORLDSTEREO_REPO = "hanshanxue/WorldStereo"
WORLDSTEREO_DEFAULT_VARIANT = "worldstereo-memory-dmd"


def _download_to_hf_cache(repo_id: str, label: str, allow: list[str] | None = None) -> bool:
    """Download into ~/.cache/huggingface/hub/ (no local_dir). Use for models
    that downstream code loads via ``from_pretrained(repo_id, ...)`` — those
    expect the HF cache layout, not a local copy."""
    api = HfApi()
    info = api.repo_info(repo_id)
    files = [s.rfilename for s in info.siblings]
    if allow:
        files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in allow)]
    if not files:
        print(f"[skip] {label}: no files match {allow}")
        return True
    print(f"[get ] {label} ({repo_id}" + (f" {allow}" if allow else "") + ") -> HF cache")
    print(f"  pulling {len(files)} file(s) sequentially ...")
    for i, fname in enumerate(files, 1):
        hf_hub_download(repo_id=repo_id, filename=fname)
        print(f"  [{i}/{len(files)}] {fname}")
    print(f"[done] {label}: HF cache")
    return True


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
    parser.add_argument("--worldstereo", action="store_true",
                        help=f"hanshanxue/WorldStereo video DiT for worldgen stage 3 (default variant "
                             f"'{WORLDSTEREO_DEFAULT_VARIANT}', ~10-25 GB). Goes to HF cache, not CKPT_DIR.")
    parser.add_argument("--worldstereo-variant", type=str, default=WORLDSTEREO_DEFAULT_VARIANT,
                        help=f"WorldStereo subfolder to pull. Default '{WORLDSTEREO_DEFAULT_VARIANT}' "
                             f"matches video_gen.py's default --model_type. Use 'worldstereo-memory' for the "
                             f"un-distilled variant, or '*' to pull all variants (large).")
    parser.add_argument("--all", action="store_true",
                        help="Download everything except --pano (use --pano explicitly because it's huge).")
    args = parser.parse_args()

    # No flag => treat as --all (mirror + pano-lora + qwen + worldstereo, but
    # NOT the 80 B --pano which the user must opt into explicitly).
    if not (args.mirror or args.pano or args.pano_lora or args.qwen or args.worldstereo or args.all):
        print("[info] no flag given; defaulting to --all (mirror + pano-lora + qwen + worldstereo). "
              "Add --pano explicitly for the 80 B full pano model.")
        args.all = True

    if args.all:
        args.mirror = True
        args.pano_lora = True
        args.qwen = True
        args.worldstereo = True

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

    if args.worldstereo:
        # Filter: keep root-level json/md + the requested variant subfolder.
        # video_gen.py needs the variant + the repo-level config files.
        variant = args.worldstereo_variant
        allow = ["*.json", "*.md", "README*", f"{variant}/*"] if variant != "*" else None
        label = f"WorldStereo ({variant})" if variant != "*" else "WorldStereo (all variants)"
        try:
            _download_to_hf_cache(WORLDSTEREO_REPO, label, allow=allow)
        except Exception as e:
            print(f"[FAIL] WorldStereo: {e}", file=sys.stderr)
            failures.append("WorldStereo")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("All requested weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
