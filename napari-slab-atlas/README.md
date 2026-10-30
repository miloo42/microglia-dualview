# napari-slab-atlas

napari widget for the acquire -> process -> register -> pick ROIs -> re-acquire loop:

1. acquire the overview (MFS4), process it, open the final fused `.zarr` in napari (drag & drop)
2. `Plugins > Slab atlas`: the zarr is picked up from the image layer; set the 540/595 and 630/700 channel indices
3. **1. Find pose** (~20 min): preprocessing, 350-start pose search, orientation test, 50 um B-spline. The 50 um
   atlas appears as soon as it is done. If the orientation is undecided: **Compare** opens a figure with both
   orientations; pick one and press **Use**.
4. **2. Refine** (~3 min): 20 um B-spline against the 10 um atlas; the layers switch to the 10 um labels
5. layers `atlas regions` (all Allen regions, hover shows acronym and name) and `selected regions` (regions
   ticked in the list, including everything below them in the tree; preset: MBO, SUM, MS, NDB, LS, RSP, ACA, AD,
   AM, AV, SUB)
6. **New** adds a `ROIs` Shapes layer; draw rectangles (same rules as before: one rectangle = full extent along its
   flat axis, overlapping rectangles in different views/slices merge into one box)
7. set the raw acquisition folder (with the pos.pos), tile overlap, views, MFS, margin; **Write pos files** writes
   `ROI<i>.pos`, `ROIs_summary.txt`, `ROIs.tsv` (for `diSPIM_utils/tools/acquire_rois.bsh`) and the shapes as
   `ROIs_shapes.csv`, by default in `<zarr name>_atlas\ROIs_pos`.
   The raw folder can also be just `pos.pos` + `AcqSettings.txt` (no .tif): then set the camera width/height (px)
   and pixel size; the camera defaults to path A (RightCam) and the scan mode to "X = scan centre" unless a
   `metadata.txt` is there

Registration results live in `<zarr name>_atlas\` next to the zarr and are loaded again when the zarr is reopened.
The registration itself is the `slab_atlas` library in `../atlas-registration`.

`pospos.py` is `diSPIM_utils/tools/roi_to_pospos.py` with the main part as a function (`write_pos_files`); the
command line still works: `python -m napari_slab_atlas.pospos <zarr> <raw_folder> <shapes.csv> [...]`.

Install in `brainglobe-env`:

```
pip install --no-deps -e "X:\OneDrive - Maastricht University\Code\microglia-dualview\atlas-registration"
pip install --no-deps -e "X:\OneDrive - Maastricht University\Code\microglia-dualview\napari-slab-atlas"
```
