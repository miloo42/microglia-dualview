"""Dry run of the plugin away from the acquisition machine: the MFS4 test slab, its existing registration
(X:\\Microglia\\Atlas\\Oct 2026 testing\\work, same layout as a <zarr>_atlas folder, nothing is recomputed) and the
Slab atlas widget, with the ROI section filled in for the test folder (pos.pos + AcqSettings.txt, no .tif).

Draw rectangles on the ROIs layer and press 'Write pos files'; the files go to Oct 2026 testing\\dry_run_pos.
The camera values (not in AcqSettings.txt) are set below; on the acquisition machine the widget reads them from
the raw .tif. Without metadata.txt the scan mode is assumed to be 'X = scan centre'.

Run in brainglobe-env:  python dry_run.py
"""
import tempfile
from pathlib import Path

import napari
import numpy as np

from napari_slab_atlas import pospos
from napari_slab_atlas._widget import SlabAtlasWidget

BASE = Path(r"X:\Microglia\Atlas\Oct 2026 testing")
ZARR = BASE / "2026_10_21_s1_540_595+630_700_MFS4_4x4_A_10ms_fused.zarr"
RESULTS = BASE / "work"
RAW = BASE                                  # pos.pos + AcqSettings.txt
# binning 4 on the 2048 x 2048 chip (launcher config), Micro-Manager pixel size 0.3625 x 4 um; matches the pos.pos
# grid: 668 um in Y and 472 um in Z = 0.9 x 742.4 um and 0.9 x 742.4 / sqrt(2) um (10 % overlap)
CAMERA = dict(width=512, height=512, pix_um=1.45)

viewer = napari.Viewer()
w = SlabAtlasWidget(viewer)
w.results_dir, w.cache_dir = RESULTS, RESULTS / "cache"
w.zarr_edit.setText(str(ZARR))
viewer.window.add_dock_widget(w, area="right", name="Slab atlas")

layers = viewer.open(str(ZARR), channel_axis=1)
for c, layer in enumerate(layers):            # one layer per zarr channel
    layer.colormap = {w.ch540.value(): "red", w.ch630.value(): "bop blue"}.get(c, "gray")
    low = np.asarray(layer.data[-1])                      # coarsest level: contrast from the tissue percentiles
    layer.contrast_limits = tuple(np.percentile(low[low > 0], (1, 99.5)))
w.open_workspace()                          # loads the existing registration and shows the atlas layers

w.raw_edit.setText(str(RAW))
w.cam_w.setValue(CAMERA["width"])
w.cam_h.setValue(CAMERA["height"])
w.pix_um.setValue(CAMERA["pix_um"])
w.pos_out.setText(str(BASE / "dry_run_pos"))
if not Path(pospos.MFS_FILE).exists():      # only on the acquisition PC: allow just the asked MFS
    mfs_fn = Path(tempfile.gettempdir()) / "MFS_X_delta_dry_run.txt"
    mfs_fn.write_text("0.5127\n")
    w.mfs_file.setText(str(mfs_fn))
    w.mfs.setValue(0.5127)
w.new_roi_layer()

# sagittal view (AP horizontal, DV vertical) in the middle of the slab thickness
n = layers[0].ndim
viewer.dims.order = tuple(range(n - 3)) + (n - 1, n - 2, n - 3)
viewer.dims.set_point(layers[0].ndim - 1, layers[0].data[0].shape[-1] // 2)
viewer.reset_view()
napari.run()
