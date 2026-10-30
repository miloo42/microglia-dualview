# atlas-registration (`slab_atlas`)

Registers a fused light-sheet slab (OME-Zarr, ~1 mm sagittal slab) to the Allen Mouse Brain Atlas CCFv3, so that
every voxel and every cell gets an Allen region. Library version of the pipeline developed in
`X:\Microglia\Atlas\Oct 2026 testing\scripts`. Method, reasons for every step and the experiment log:
[docs/method.md](docs/method.md). Its script names map to functions here (table below).

Install (in `brainglobe-env`, which has itk-elastix, brainglobe-registration, napari):

```
pip install --no-deps -e "X:\OneDrive - Maastricht University\Code\microglia-dualview\atlas-registration"
```

## Use

From a script or notebook:

```python
from slab_atlas import Workspace, find_pose, refine_20um
from slab_atlas import qc, export

ws = Workspace(r"X:\...\fused\sample_fused.zarr", ch_540=0, ch_630=1)   # results: sample_fused_atlas\ next to the zarr
find_pose(ws)          # prepare + pose search + ranking + orientation test + 50 um B-spline (~20 min)
refine_20um(ws)        # 20 um B-spline vs the 10 um atlas, 10 um labels in image space (~2 min)
qc.figures(ws); export.export_spaces(ws)
cells = export.map_cells(ws, zyx_level0_voxels)       # DataFrame with region id/acronym and CCF coordinates
```

Command line: `python -m slab_atlas {find-pose,choose,refine,all,scores,figures,export,cells,test} <zarr> [...]`
(`--help` for options).

Steps that already have results are skipped by `find_pose`. When the orientation test is undecided, compare both
orientations (`qc.compare_50um(ws)`) and pick one: `choose_pose(ws, flip=0 or 1)`, then `refine_20um(ws)`.

| function | old script | does |
|---|---|---|
| `pipeline.prepare` | 01_prepare_slab | 50 um registration images (bright objects removed at ~16 um) + tissue mask |
| `pipeline.pose_search` | 02_pose_search | 350 rigid/similarity/constrained-affine runs from different start poses |
| `pipeline.rank_poses` | 03_rank_poses | final pose and deterministic score of each run |
| `pipeline.refine_50um` | 04_refine_50um | B-spline at 50 um of one ranked run |
| `pipeline.orientation_test` | 05_orientation_test | which slab face is medial |
| `pipeline.choose_pose` | (04 on rank 1) | best run of the chosen orientation -> `03_reg_50um` |
| `Atlas.ensure_cache` | 06_cache_10um | 10 um atlas -> .npy, once |
| `pipeline.refine_20um` | 07_refine_20um | B-spline at 20 um vs the 10 um atlas, warped 10 um labels |
| `qc.stage_scores` | 08_stage_scores | scores per registration stage |
| `qc.figures`, `qc.compare_50um` | 09_figures, qc_compare | figures |
| `export.export_spaces` | 10_export_spaces | TIFFs in image and atlas space |
| `export.map_cells`, `export.self_test` | 11_map_points | cells -> regions |

Modules: `atlas.py` (Allen files, region tree), `slab.py` (zarr reading, slab grids, preprocessing),
`elastix.py` (itk-elastix helpers), `pipeline.py`, `qc.py`, `export.py`.

Generalised from the scripts: voxel size, shape and pyramid levels are read from the zarr metadata (the
preprocessing level is the one closest to 16 um, opening disks are given in um), channel order is a setting, and
all paths are per workspace. Checked on `2026_10_21_s1_..._fused.zarr`: preprocessing images and pose ranking
identical to the scripts, 20 um labels identical in 100 % of tissue voxels, self-test 14/14.

## Files

- Allen files: `X:\Microglia\Atlas\Allen` (`Atlas(atlas_dir=...)` / `Workspace(atlas_dir=...)` to change). The 10 um
  cache (6.8 GB) goes to `X:\Microglia\Atlas\Allen\npy_cache`, built on first use (~1 min); `cache_dir=` to change.
- Results per slab in `<zarr name>_atlas\`: `workspace.json` (settings), `01_prep`, `02_pose_search`,
  `03_reg_50um`, `04_reg_20um` (final transforms, `warped_annot20.npy`, `export\`), `figures`.
