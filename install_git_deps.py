"""Install git+ dependencies from requirements_git.txt without uv pip or pip's
VCS handler, both of which fail on this stack:

- pip's VCS handler hits "RuntimeError: cannot join thread before it is started"
  (Python 3.12 + Windows subprocess._communicate thread race).
- uv pip --no-build-isolation runs builds in subprocesses that don't reliably
  inherit CUDA_HOME on Windows — the build sees "CUDA_HOME not set" even when
  it's exported in the parent shell.

This script: git-clones each git+URL to .third_party_clones/<repo>, then runs
`pip install --no-build-isolation <localdir>` (plain pip, which inherits env
normally) one dep at a time.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CLONES_DIR = REPO_ROOT / ".third_party_clones"
REQUIREMENTS = REPO_ROOT / "requirements_git.txt"
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

URL_RE = re.compile(r"git\+(?P<url>[^@\s]+?)(?:@(?P<ref>[^\s#]+))?(?:#egg=(?P<egg>[^\s]+))?\s*$")

# Force CUDA_HOME into the subprocess env every time — pip's build subprocess
# doesn't reliably inherit it on Windows + Python 3.12. Pin to the v12.8 toolkit
# (matches torch 2.7+cu128 wheels).
CUDA_DEFAULT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
CUDA_HOME = os.environ.get("CUDA_HOME") or CUDA_DEFAULT
# Mutate the actual process env so EVERY spawned subprocess (including pip's
# nested _in_process subprocess that runs the build backend) inherits these.
# Passing env= to subprocess.call only worked for direct children; pip's nested
# spawn was still seeing an empty CUDA_HOME on Windows + Py 3.12.
os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["CUDA_PATH"] = CUDA_HOME

# spz uses scikit-build-core which hits the Python 3.12 + Windows
# subprocess._communicate thread race ("cannot join thread before it is
# started") when discovering cmake. There's no workaround short of downgrading
# Python — mark it OPTIONAL so a failure here doesn't fail the whole step.
OPTIONAL = {"spz"}

# Repo dir name -> python import name (for "already installed?" skip).
IMPORT_NAME = {
    "fused-ssim": "fused_ssim",
    "MoGe": "moge",
    "pytorch3d": "pytorch3d",
    "spz": "spz",
    "nerfview": "nerfview",
}


def is_installed(py: Path, modname: str) -> bool:
    rc = subprocess.call(
        [str(py), "-c", f"import {modname}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


def main() -> int:
    if not REQUIREMENTS.exists():
        print(f"FATAL: {REQUIREMENTS} not found", file=sys.stderr)
        return 2
    if not PY.exists():
        print(f"FATAL: {PY} not found (run setup.bat steps 1-3 first)", file=sys.stderr)
        return 2

    CLONES_DIR.mkdir(parents=True, exist_ok=True)

    git_lines = [
        ln.strip()
        for ln in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith("git+")
    ]
    print(f"Found {len(git_lines)} git+ requirement(s).")
    print()

    failures: list[str] = []
    for i, line in enumerate(git_lines, 1):
        m = URL_RE.match(line)
        if not m:
            print(f"[{i}/{len(git_lines)}] SKIP malformed: {line}")
            continue
        url = m.group("url")
        ref = m.group("ref")
        name = url.rstrip(".git").rstrip("/").split("/")[-1]
        dest = CLONES_DIR / name

        print(f"[{i}/{len(git_lines)}] {name} ({url}{'@' + ref if ref else ''})")

        modname = IMPORT_NAME.get(name)
        if modname and is_installed(PY, modname):
            print(f"  [skip] already importable as `{modname}`")
            continue

        # Clone or update.
        if not dest.exists():
            print(f"  cloning -> {dest}")
            rc = subprocess.call(["git", "clone", "--quiet", url, str(dest)])
            if rc != 0:
                failures.append(f"{name}: git clone failed (rc={rc})")
                continue
        else:
            print(f"  reusing existing clone {dest}")
            subprocess.call(["git", "-C", str(dest), "fetch", "--quiet"])

        if ref:
            rc = subprocess.call(["git", "-C", str(dest), "checkout", "--quiet", ref])
            if rc != 0:
                failures.append(f"{name}: git checkout {ref} failed (rc={rc})")
                continue

        # Build env: explicitly inject CUDA_HOME / CUDA_PATH every time. pip's
        # build subprocess on Windows + Py3.12 doesn't reliably inherit them.
        build_env = os.environ.copy()
        build_env["CUDA_HOME"] = CUDA_HOME
        build_env["CUDA_PATH"] = CUDA_HOME

        # --no-deps: skip pip's recursive dep resolution. Some git+ packages
        # (e.g. MoGe -> utils3d) pin transitive git+ URLs and pip's VCS handler
        # triggers a Python 3.12 + Windows subprocess._communicate thread race
        # ("cannot join thread before it is started"). The deps we actually need
        # are already in requirements.txt or installed separately (utils3d 0.1.3).
        print(f"  pip install --no-build-isolation --no-deps .  (CUDA_HOME={CUDA_HOME})")
        rc = subprocess.call(
            [str(PY), "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(dest)],
            cwd=str(dest),
            env=build_env,
        )
        if rc != 0:
            tag = "(OPTIONAL)" if name in OPTIONAL else ""
            failures.append(f"{name}: pip install failed (rc={rc}) {tag}")

    print()
    blocking = [f for f in failures if "(OPTIONAL)" not in f]
    optional = [f for f in failures if "(OPTIONAL)" in f]
    if optional:
        print(f"OPTIONAL FAILURES ({len(optional)}) — not blocking:")
        for f in optional:
            print(f"  {f}")
    if blocking:
        print(f"FAILED ({len(blocking)}):")
        for f in blocking:
            print(f"  {f}")
        return 1
    n_ok = len(git_lines) - len(failures)
    print(f"All {n_ok} required git+ deps installed (+ {len(optional)} optional failures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
