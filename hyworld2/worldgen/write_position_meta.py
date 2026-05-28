"""Generate position_meta_info.json from a 3DGS gaussians.ply.

Computes a sensible default camera pose: looking at the scene centroid from
a distance proportional to the scene's bounding-box diagonal. Useful when
show_gs.py refuses to render (black screen) because the default
position_meta_info.json placed the camera at the origin — which is inside or
far from the actual Gaussian cloud.

Usage:
    python write_position_meta.py path/to/gaussians.ply
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python write_position_meta.py <gaussians.ply>")
        sys.exit(2)
    ply_path = Path(sys.argv[1])
    if not ply_path.is_file():
        print(f"ERROR: not a file: {ply_path}")
        sys.exit(1)

    print(f"Reading {ply_path} ...")
    pd = PlyData.read(str(ply_path))
    vx = pd["vertex"]
    pts = np.stack([vx["x"], vx["y"], vx["z"]], axis=-1).astype(np.float64)
    print(f"  N gaussians: {pts.shape[0]:,}")

    centroid = pts.mean(axis=0)
    bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    print(f"  centroid : {centroid.tolist()}")
    print(f"  bbox min : {bbox_min.tolist()}")
    print(f"  bbox max : {bbox_max.tolist()}")
    print(f"  diagonal : {diag:.3f}")

    # Place camera 1.0× diagonal away from centroid, offset along +Z so we
    # look at the scene from outside. Use +Y up (OpenGL convention).
    offset = np.array([0.0, 0.0, diag * 1.0], dtype=np.float64)
    camera_pos = centroid + offset
    look_at = centroid

    meta = {
        # show_gs.py reads these as initial camera params:
        #   server.initial_camera.position = center_point
        #   server.initial_camera.look_at  = facing_direction
        #   server.initial_camera.up       = up_direction
        "up_direction": [0.0, 1.0, 0.0],
        "facing_direction": look_at.tolist(),
        "center_point": camera_pos.tolist(),
    }

    out_path = ply_path.parent / "position_meta_info.json"
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
