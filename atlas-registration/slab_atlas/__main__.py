"""Command line: python -m slab_atlas <step> <fused.zarr> [options]

steps:
  find-pose     prepare + pose search + ranking + orientation test + 50 um B-spline (skips steps with results)
  choose        use the best run of one orientation as the 50 um result (--flip 0|1, default: orientation test)
  refine        B-spline at 20 um vs the 10 um atlas, 10 um labels in image space
  all           find-pose + refine
  scores        score every registration stage
  figures       step-by-step figures (+ orientation comparison)
  export        TIFFs: atlas in image space and image in atlas space
  cells CSV     Allen region per cell (CSV columns z,y,x in level-0 voxels) -> <CSV>_regions.csv
  test          one point per major region, checks both mapping routes
Results go to <zarr name>_atlas/ next to the zarr (or --out)."""
import argparse
from pathlib import Path

import pandas as pd

from . import pipeline, qc, export


def main():
    ap = argparse.ArgumentParser(prog="python -m slab_atlas", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["find-pose", "choose", "refine", "all", "scores", "figures", "export", "cells", "test"])
    ap.add_argument("zarr")
    ap.add_argument("csv", nargs="?", help="cells CSV (step 'cells')")
    ap.add_argument("--out", help="results folder (default <zarr name>_atlas next to the zarr)")
    ap.add_argument("--ch-540", type=int, help="channel index of the 540/595 (autofluorescence) channel (default 0)")
    ap.add_argument("--ch-630", type=int, help="channel index of the 630/700 (microglia) channel (default 1)")
    ap.add_argument("--atlas-dir", help=r"Allen files (default X:\Microglia\Atlas\Allen)")
    ap.add_argument("--cache-dir", help="10 um atlas .npy cache (default <atlas-dir>/npy_cache)")
    ap.add_argument("--workers", type=int, default=5, help="parallel pose-search processes")
    ap.add_argument("--flip", type=int, choices=[0, 1], help="orientation: 0 = as acquired, 1 = flipped")
    a = ap.parse_args()
    ws = pipeline.Workspace(a.zarr, a.out, a.ch_540, a.ch_630, a.atlas_dir, a.cache_dir)
    if a.step == "find-pose":
        pipeline.find_pose(ws, workers=a.workers, flip=a.flip)
    elif a.step == "choose":
        pipeline.choose_pose(ws, a.flip)
    elif a.step == "refine":
        pipeline.refine_20um(ws)
    elif a.step == "all":
        pipeline.run_all(ws, workers=a.workers, flip=a.flip)
    elif a.step == "scores":
        qc.stage_scores(ws)
    elif a.step == "figures":
        qc.figures(ws)
        qc.compare_50um(ws)
    elif a.step == "export":
        export.export_spaces(ws)
    elif a.step == "cells":
        res = export.map_cells(ws, pd.read_csv(a.csv)[["z", "y", "x"]].to_numpy(float))
        o = Path(a.csv).with_name(Path(a.csv).stem + "_regions.csv")
        res.to_csv(o, index=False)
        print("wrote", o, "| routes agree for %.1f%% of cells" % (100 * (res.region_id == res.region_id_ccf).mean()))
    elif a.step == "test":
        export.self_test(ws)


if __name__ == "__main__":
    main()
