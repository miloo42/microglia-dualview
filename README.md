# microglia-dualview

Processing of dual-view light-sheet data of microglia: BigStitcher view registration and export (Fiji macros),
per-view GPU deconvolution and geometric-mean fusion, and registration of the fused FOV to the lowres overview.

- `bigstitcher-macros-singlechannel/`, `bigstitcher-macros-multichannel-multiview/`: headless Fiji macros
- `deconvolution/dualview_deconv.py`: library, chunked Richardson-Lucy per view (tnia, CuPy, dask) and `fuse_gmean`
- `fov-registration/fov_registration.py`: library, phase correlation of the fused FOV to the lowres overview;
  `view_registered_fovs.py` shows the result in napari
- `atlas-registration/`: library `slab_atlas`, registration of the fused slab to the Allen atlas CCFv3
- `napari-slab-atlas/`: napari plugin, runs the atlas registration, shows the regions, writes pos.pos files for ROIs
  drawn on the zarr
- the libraries are called from a notebook in each data folder (e.g. `X:\Microglia\2026_10_24_13bF1\process_13bF1.ipynb`),
  run in the `decon-env` conda environment

## Development notes

### Background subtraction before deconvolution (not implemented)

There is no background subtraction before deconvolution. The views keep a constant floor: in
2026_10_24_13bF1, nonzero voxels of view A and view B have a 0.1 % / 1 % / 5 % percentile of about 120 / 135 / 147
and a median of 171. This floor is camera offset plus some autofluorescence. Richardson-Lucy treats it as signal,
which reduces the contrast gain and can add ringing near bright structures.

Guo et al. 2020 (Nat Biotechnol 38, 1337, "Rapid image deconvolution and multiview fusion for optical microscopy";
code: regDeconProject) subtract "a uniform background with intensity equivalent to the average of 100 dark
(no excitation light) background images before registration and/or deconvolution", and replace zeros with a small
value (0.001).

Plan: add a `background=` argument to `dualview_deconv.deconvolve` that subtracts a constant before RL. Voxels inside
the view's coverage that end up <= 0 are set to a small positive value. Voxels outside the coverage stay 0, so the
`img_chunk > 0` mask keeps working. Use the camera dark level measured from dark frames; a low percentile of the
data is a fallback but also removes real autofluorescence.

### Intensity normalization between views (not needed)

Guo et al. 2020 do not normalize intensities between views. A global brightness difference between A and B only
scales the geometric-mean fusion by a constant. In 2026_10_24_13bF1 both views have the same median (171).
Depth-dependent attenuation within a view is not corrected either.

### Joint deconvolution instead of per-view RL + geometric mean (alternative)

Guo et al. deconvolve both views jointly: one estimate, initialized as (A + B) / 2, updated alternately with view A
(PSF A) and view B (PSF B) in every iteration (`WBDeconvolution/DeconDualView.m`). For large cleared-tissue
volumes they do this in 640^3 subvolumes and crop 45 px from each edge
(`ClearedTissueProcessing/BigData_Postprocessing.m`). The current pipeline instead deconvolves each view on its
own and fuses by geometric mean.
