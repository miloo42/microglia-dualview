# Registering a light-sheet brain slab to the Allen Mouse Brain Atlas (CCFv3)

This folder holds a pipeline that maps a cleared, light-sheet-imaged ~1 mm sagittal brain slab onto the Allen Common
Coordinate Framework v3, so that every detected cell can be given an Allen brain region. It also documents every
experiment that led to the final design, including the ones that failed, and why each step is there.

Test sample: `2026_10_21_s1_540_595+630_700_MFS4_4x4_A_10ms_fused.zarr` (E4FAD mouse, two channels).

---

## 1. Result in one paragraph

The slab sits about **1.1 mm lateral of the midline** (slab centre), tilted by **~11° of yaw** (rotation about the
dorsoventral axis) and **~5° of roll** (about the anteroposterior axis). Because of the yaw, its tissue spans from about
0.5 mm across the midline at the hindbrain end to about 2.8 mm lateral at the olfactory end. The face of the slab with
the higher x index in the image is the medial one ("flipped" relative to the first assumption). The final transform
(rigid start → similarity → constrained affine → B-spline at 50 µm → B-spline at 20 µm against the 10 µm atlas) aligns
the tissue outline with Dice 0.981 and improves the anatomical intensity match at every stage (table in §5). Cells are
labelled by looking their position up in the 10 µm Allen annotation warped into image space.

To label cells: `python scripts/11_map_points.py cells.csv` (columns `z,y,x` in full-resolution voxels).

---

## 2. Data and conventions

| item | value |
|---|---|
| image | OME-Zarr, 4 pyramid levels; level 0 = 4.1012 µm isotropic, shape (t, c, z, y, x) = (1, 2, 3301, 1863, 379) |
| channel 0 | 540/595 nm: tissue autofluorescence plus very bright blobs (likely amyloid plaques), present only in some regions |
| channel 1 | 630/700 nm: microglia, evenly distributed, little anatomical contrast otherwise |
| image axes | z = anterior → posterior, y = dorsal → ventral, x = slab thickness (left–right). Same order as the atlas, so no transposition is needed |
| atlas | Allen CCFv3 2017: `average_template_{10,50}.nrrd`, `annotation_{10,50}.nrrd` in `X:\Microglia\Atlas\Allen` |
| atlas axes | (AP, DV, ML); midline at ML = 5700 µm |
| region tree | `Allen/structures_dl/structure_graph_1.json` (downloaded from the Allen API) |
| grids | atlas voxel *m* at *res* µm sits at CCF position *res·m* µm (checked: 10 µm and 50 µm volumes agree to r = 0.999 with this convention). Slab grid with voxel size *u*: voxel *i* ↔ level-0 position *(i + 0.5)·u/4.1012 − 0.5* |
| software | Python env `brainglobe-env` (itk-elastix 5.4, brainglobe-registration, scipy, scikit-image, zarr, pynrrd) |

---

## 3. Pipeline: what each step does and why

Run from `scripts/` in `brainglobe-env`. Times are for this slab on this workstation (20 cores).

| # | script | time | what it does |
|---|---|---|---|
| 1 | `01_prepare_slab.py` | 10 s | build the 50 µm registration images and the tissue mask |
| 2 | `02_pose_search.py` | 10 min | 350 rigid/affine registrations from different starting poses |
| 3 | `03_rank_poses.py` | 20 s | measure the final pose of each run and rank the runs |
| 4 | `04_refine_50um.py` | 30 s | B-spline at 50 µm on the best pose |
| 5 | `05_orientation_test.py` | 6 min | decide which slab face is medial |
| 6 | `06_cache_10um.py` | once, 1 min | unpack the 10 µm atlas to fast `.npy` files |
| 7 | `07_refine_20um.py` | 2 min | fine B-spline at 20 µm against the 10 µm atlas; warp the 10 µm labels into image space |
| 8 | `08_stage_scores.py` | 40 s | score every stage on the same grid |
| 9 | `09_figures.py` | 30 s | step-by-step figures |
| 10 | `10_export_spaces.py` | 30 s | TIFF exports: atlas in image space and image in atlas space |
| 11 | `11_map_points.py` | seconds | region + CCF coordinates for each cell |

Shared code: `common.py` (paths, constants, region tree), `prep.py` (preprocessing and slab grid geometry),
`reg.py` (elastix helpers, transform evaluation, inverse-free mapping). `qc_compare.py` draws two registrations side by side.

### 3.1 Preprocessing (`01_prepare_slab.py`, `prep.py`)

**What.** Load pyramid level 2 (16.4 µm). In each sagittal plane, remove bright objects with a grey-level morphological
opening (disk, 150 µm for 540, 115 µm for 630), then median and light Gaussian smoothing. Resample exactly onto the 50 µm
grid, normalise to the 1st–99th percentile inside the tissue. Tissue mask: Otsu threshold on the smoothed 630 channel,
cleaned up morphologically.

**Why.**
* The Allen template is an *autofluorescence* image: grey matter is moderately bright, white matter (corpus callosum,
  arbor vitae, fimbria) is dark. Registration needs an image with the same kind of contrast.
* The **540 channel background** is the most template-like image: cortex vs. white matter, olfactory bulb and hippocampus
  are all visible. Its bright blobs are biological signal that exists only in some regions and has no counterpart in the
  atlas, so they must go (`work/archive/diag_channels_16um.png`).
* The **630 channel** is almost flat apart from microglia and a bright, contrast-inverted fimbria; on its own it gave the
  optimiser almost nothing to work with (experiment v0 below). It is kept as a second, weaker image.
* Blobs are removed **at 16 µm, before downsampling**: at 33–50 µm the plaques are already smeared into the background
  and can no longer be separated from it.
* A **disk**-shaped opening, not a square one: the square left blocky edges that acted as false structure at 20 µm.
* The opening keeps the background and removes only objects smaller than the disk; blobs larger than ~150 µm survive as
  soft spots (visible in `figures/fig1_preprocessing.png`), an accepted limitation.

### 3.2 Scale and starting geometry

* The sample was cleared, so the physical size of the tissue is uncertain (a 70 % shrinkage was suspected). Rather than
  assume it, the scale was measured: tissue extents at the nominal 4.1 µm voxel size are already close to CCF size, and the
  registration settles on a CCF/slab size ratio of about 1.1–1.25 (i.e. ~80–90 %, not 70 %). The search starts at 1.15 and
  the similarity stage finds the final value.
* Image axes match atlas axes (checked visually), so only the thickness direction can be mirrored (see orientation).

### 3.3 Pose search (`02_pose_search.py`, `03_rank_poses.py`)

**What.** 350 registrations, each starting from a different hypothesis:
which face is medial (2) × medial edge 0, 250, 500, 750, 1000 µm from the midline (5) × start yaw −30…30° (7) ×
start roll −20…20° (5). Each run: Euler start → similarity → constrained affine, 3 resolution levels, ~6 s, 5 runs in
parallel. Each run is then scored *deterministically on all voxels*: normalised mutual information (NMI) of the 540 and
630 images against the warped template inside the tissue, plus the outline Dice. Runs are ranked by the sum of z-scores.

**Why a search and not a single registration.** The slab's position is genuinely unknown: the medial face can be 0–1 mm
from the midline, the slab can be rotated by up to ~20° on any axis, and which face is medial cannot be known in advance.
Gradient-based registration only finds the nearest optimum; in the test runs the final position stayed close to where it
started (see `figures/fig2_rough_positioning.png`, right). Starting from many poses and keeping the best is the standard
remedy and costs minutes at 50 µm. No grid search over elastix settings was needed.

**Design choices inside each run, and why:**
* **Three metrics.** Mattes MI on the 540 image (weight 0.4), MI on the 630 image (0.25) and mean squares between the
  smoothed tissue mask and the smoothed atlas brain mask (the *outline term*, 0.35), with elastix's relative weighting so
  the gradients are balanced. MI alone was too weak with this contrast (v0); the outline term is modality independent.
* **Outline term only on the natural brain surface.** The slab's medial and lateral faces are *cuts*: the atlas brain
  continues beyond them. An outline term that penalises atlas tissue beyond the cut faces biases the pose (bug found in
  v1). For each line through the slab thickness, only voxels between the first and last tissue voxel count; lines with
  no tissue (outside the brain in-plane) count within 400 µm. See `work/01_prep/outline_mask.png`.
* **Constrained affine.** A free 12-parameter affine *collapses* a thin slab: it squeezes the thickness (scale 0.16 in one
  test) or shears it so the whole slab maps onto one well-matching atlas plane (`work/archive/v0_free_affine_collapse.png`).
  The affine stage therefore uses elastix's `AffineDTITransform` with the thickness scale and all shears frozen; rotations,
  translations and in-plane (AP/DV) scales stay free.
* **Default parameter maps.** brainglobe-registration's `setup_parameter_object` *clears* elastix's defaults before adding
  yours, which silently drops required keys (registration type, pyramids, resampler); this explains the "does not exist"
  warnings of the January attempt. Here every map starts from elastix's defaults.
* **Deterministic scoring.** Elastix's final metric is computed on a fresh random sample each iteration and is not
  comparable between runs (one reason the January grid search was misleading). Scores are therefore recomputed on all
  voxels after each run.

**Result.** The top 15 runs, started from offsets 0–1000 µm and different rolls, all converge to the same pose within
±11 µm and ±0.7°. The best run of the other orientation lands on the same position and tilt, differing only in which face
is medial.

### 3.4 Which face is medial (`05_orientation_test.py`)

**What.** Refine the top 5 runs of each orientation through the 50 µm B-spline and compare final scores. Decide for the
orientation favoured by every score whose difference exceeds twice the scatter between runs; ignore ties.

**Why.** It cannot be determined beforehand, and the multistart score alone is weak evidence: before the B-spline the
intensity scores slightly favour one orientation and the outline Dice the other. After the B-spline, the outline fits
equally well in both orientations (Dice is a tie), while the internal anatomy separates them.

**Result for this slab:** see §5. Verdict "flipped", which also matched the visual check
(`figures/qc_compare_03_reg_50um_vs_refine_rank16.png`).

### 3.5 B-spline at 50 µm (`04_refine_50um.py`)

B-spline with a final control-point spacing of 0.6 mm (schedule 2.4 → 1.2 → 0.6 mm), the same three metrics plus a
bending-energy penalty (relative weights 0.35 / 0.2 / 0.25 / 0.2). Corrects tissue deformation from clearing and mounting
that a linear transform cannot.

### 3.6 Fine B-spline at 20 µm against the 10 µm atlas (`07_refine_20um.py`)

**What.** Slab images rebuilt on a 20 µm grid with the same preprocessing; the 10 µm template is block-averaged to 20 µm
and cropped to the slab plus 1 mm. B-spline with a final spacing of 0.25 mm, initialised with the 50 µm result. Labels
are then read from the **10 µm annotation** at each slab voxel's mapped position (nearest neighbour), so no label detail is
lost.

**Why 20 µm and not 10 µm.**
* After plaque removal and smoothing, the slab images carry structure at roughly 50–150 µm; a 10 µm grid adds voxels but
  no information.
* The detail of the deformation is set by the control-point spacing (250 µm), not the voxel size, and elastix evaluates
  the metric on 8192 random points per iteration either way.
* The expected registration error (~50–150 µm: tissue deformation, residual plaques, contrast differences) is far above
  10 µm.
* 10 µm would need ~8× the memory (more than the free RAM) and several times the run time.
* Labels and cell coordinates still use the 10 µm annotation directly.

### 3.7 Labelling cells (`11_map_points.py`)

Cells are labelled in **image space**: each cell's full-resolution position is converted to the 20 µm slab grid and looked
up in `atlas_annotation_in_slab.tif` (the 10 µm annotation warped onto the slab). This needs only the forward transform.
The script also maps each cell through the transform with transformix and looks it up in the 10 µm annotation as an
independent check, and reports CCF coordinates in µm. Self-test (`--test`, one point per major region): 14/14 in the
expected region, both routes agree 14/14.

### 3.8 Image in atlas space and the inverse transform (`10_export_spaces.py`)

Registration produces the transform *slab → CCF*. Warping the atlas onto the slab only needs this direction (for each slab
voxel, look up where it lands in the atlas). Warping the *slab into atlas space* needs the opposite: for each atlas voxel,
the slab position that maps onto it.

* A linear transform is inverted by inverting its matrix.
* A B-spline transform, *T(x) = x + Σ cₖ β(x − xₖ)*, has no closed-form inverse: the displacement at *x* depends on *x*
  through the B-spline basis functions. The inverse is found numerically, point by point, by solving *T(x) = y* for *x*.
* Method used: the forward map is already known at every 20 µm slab voxel (`slab20_to_ccf_um.npy`). For each atlas voxel
  *y*, start at the slab voxel whose image is nearest to *y* (k-d tree), then apply Newton steps
  *x ← x + J⁻¹ (y − T(x))*, with *J* the Jacobian of the map from finite differences. Residual after two steps: below
  0.01 µm for all 9.8 million atlas voxels covered by the slab.
* This only works if the transform does not fold (each atlas point has exactly one pre-image). Checked: the Jacobian
  determinant has the same sign at all 7.5 million tissue voxels (negative everywhere because the flipped orientation
  includes a mirror; |det| from 0.12 to 2.8, median 1.28 = volume scale).
* Alternatives with the same goal: elastix's own inversion (register the fixed image to itself with a
  `DisplacementMagnitudePenalty` metric, using the forward transform as the initial transform), or fixed-point
  iteration on displacement fields (ITK `InverseDisplacementFieldImageFilter`).

---

## 4. Outputs (`work/`)

| folder | contents |
|---|---|
| `cache/` | 10 µm template and annotation as `.npy` (6.8 GB, regenerate with step 6) |
| `01_prep/` | 50 µm slab images (`slab50_F_540_bg`, `slab50_G_630_bg`, `slab50_raw`, `slab50_mask`), `qc_prepare.png`, `outline_mask.png` |
| `02_pose_search/` | `multistart.json` (all 350 runs), `multistart_ranked.json`, `multistart_summary.png`, `orientation_test.json`, `refine_rank16/` (best run of the other orientation, for comparison) |
| `03_reg_50um/` | chosen pose + 50 µm B-spline: `TransformParameters.0–3.txt`, `geometry.json`, warped template/annotation (50 µm) |
| `04_reg_20um/` | **final result**: `TransformParameters.0–4.txt` (apply in order), `geometry.json`, `warped_annot20.npy`, `slab20_to_ccf_um.npy`, `stage_scores.json` |
| `04_reg_20um/export/` | TIFFs with 20 µm voxel size. Image space: `slab_540/630.tif`, `atlas_template_in_slab.tif`, `atlas_annotation_in_slab.tif` (uint32 Allen ids). Atlas space: `slab_540/630_in_atlas.tif`, `slab_mask_in_atlas.tif`, `atlas_template20.tif`, `atlas_annotation20.tif`. `structures.csv` (id → acronym → name), `qc_atlas_space.png` |
| `figures/` | `fig1_preprocessing`, `fig2_rough_positioning`, `fig3_steps_overview`, `fig4_final_detail`, orientation comparison |
| `archive/` | small images and score files from superseded experiments, referenced in §6 |

Region-id TIFFs are plain 32-bit integer TIFFs (Allen ids exceed the ImageJ format): open them as a Labels layer in
napari, or with Bio-Formats in Fiji.

`geometry.json` records what is needed to use the transforms outside these scripts: `flip` (if 1, the transform is defined
on the slab mirrored along x: x′ = 30 − x on the 50 µm grid), the fixed-image spacing (50 µm grid spacing including the
start scale) and the moving-image origin (CCF origin in elastix's physical frame).

---

## 5. Final numbers for this slab

Every stage scored on the same 20 µm grid against the 10 µm template (`08_stage_scores.py`):

| stage | outline Dice | NMI 540 | NMI 630 |
|---|---|---|---|
| start pose (best multistart start) | 0.779 | 1.0030 | 1.0055 |
| similarity | 0.904 | 1.0084 | 1.0065 |
| constrained affine | 0.936 | 1.0106 | 1.0070 |
| B-spline 50 µm | 0.975 | 1.0166 | 1.0073 |
| B-spline 20 µm | **0.981** | **1.0198** | **1.0083** |

NMI here is Studholme's normalised MI (1 = independent, 2 = identical). Across modalities the absolute values are small;
the differences between stages and between candidates are what matter.

Orientation test (top 5 runs per orientation after the 50 µm B-spline; mean ± SD):

| score | as acquired | flipped | gap ÷ scatter |
|---|---|---|---|
| outline Dice | 0.9816 ± 0.0002 | 0.9813 ± 0.0001 | −1.9 (tie, ignored) |
| NMI 540 | 1.0149 ± 0.0001 | **1.0171 ± 0.0001** | **+17.6** |
| NMI 630 | 1.0078 ± 0.0001 | **1.0083 ± 0.0001** | **+4.1** |

Verdict: **flipped** (high-x face medial).

Pose: slab centre ~1.1 mm lateral, yaw ~11°, roll ~−5°, CCF/slab size ratio ~1.1 in-plane.

---

## 6. Experiment log

**January 2026: grid search (failed).** 72 affine runs over resolutions, iterations, sample counts, histogram bins and
sampler type. Every run "succeeded", but the ranking was meaningless:
(1) the score was elastix's final metric, a single noisy random-sample value, not comparable between runs with different
pyramid depths; (2) runs had not converged (the metric was still improving after 2000 iterations; step size never
decayed), so the image "ran away"; (3) the mask was multiplied into the image instead of being passed to elastix, so the
zero background still dominated MI; (4) brainglobe's parameter setup dropped required elastix keys; (5) the varied
settings were not the ones that mattered (step size, initial position, masks).

**v0: single start at the midline, 630 channel (Oct 2026).** New scripts, 50 µm atlas, SimpleITK first, then itk-elastix
once the environment was fixed. Findings:
* the free affine collapses the slab; the constrained affine (`AffineDTITransform`, thickness scale and shears frozen) fixed it;
* adding the outline term raised Dice from 0.63 to 0.69, and the B-spline to 0.75;
* five preprocessing variants of 630/540 all ended within Dice 0.736–0.753 (`archive/v0_preprocessing_variant_scores.json`):
  preprocessing mattered less than the transform model;
* restarts within ±150 µm and scale 1.0–1.2 gave the same regions (Dice 0.85–0.98 for large regions).
* **Rejected on inspection:** the outline matched but internal anatomy did not (`archive/v0_midline_start_630_regions.png`).
  The single start assumed the medial face was at the midline; the stability test only probed ±150 µm around that guess.

**Diagnosis.** Viewing both channels at 16 µm next to the template showed that the 630 channel has almost no
autofluorescence-like contrast, while the 540 background does (`archive/diag_channels_16um.png`,
`archive/diag_template_sagittal.png`). Preprocessing moved to the 540 background (plaques removed at 16 µm) plus 630.

**v1: multistart, 250 starts.** Converged tightly, but the best pose placed the medial face 1.5 mm from the midline and
conflicted with the tissue seen at one face. Comparing candidates revealed the **cut-face bug** in the outline term
(penalising atlas tissue beyond the slab's cut faces). `archive/v1_*`.

**v2: corrected outline term, yaw range ±30°, 350 starts.** Both orientations now converge to the same position and tilt.
Orientation decided by the refined-score test (the multistart score alone was dominated by Dice).

**10 µm refinement.** 20 µm B-spline against the 10 µm template; labels from the 10 µm annotation. Every score improved
over the 50 µm result.

**Square → disk opening.** The square structuring element left blocky edges visible at 20 µm
(`archive/v2_orientation_candidates_square_opening.png` was made before the change). Switched to a disk + smoothing and
re-ran everything downstream; scores equal or slightly better.

**Clean-up.** Work folder reorganised by pipeline step; all downstream steps re-run from the saved multistart results to
confirm identical results.

---

## 7. Limitations

* **Accuracy.** Expect roughly 50–150 µm. Assignments to large and mid-sized regions are reliable; thin layers and small
  nuclei near borders are not.
* **Cut faces.** Atlas regions just beyond the slab's faces have no tissue; cells cannot be there, so counts are not
  affected, but region *volumes* sampled by the slab should be computed from the warped annotation inside the tissue mask.
* **Residual plaques.** Blobs larger than ~150 µm survive the opening and can locally attract the registration.
* **Caudal brainstem** extends beyond the posterior edge of the CCF volume and receives no label.
* **Thickness scale** is tied to the in-plane scale (frozen in the affine stage); an axial scale error of the light-sheet
  (refractive-index mismatch) would shift assignments across the slab thickness slightly.
* **Orientation** is decided by data; on a slab where the two clear scores disagree the test returns "undecided" and the
  slab needs a visual check (`qc_compare.py`).

---

## 8. Running on a new slab

1. Set `ZARR` in `scripts/common.py`.
2. Check that the image axes match the atlas (z = AP, y = DV, x = thickness) and that channel 0 is the 540 channel.
3. Run steps 1–5, look at `02_pose_search/multistart_summary.png` (do the top runs agree?) and the orientation verdict,
   then steps 7–10 (step 6 only once) and check `figures/fig3_steps_overview.png`.
4. Label cells with step 11.

Archived scripts in `scripts/archive/` (single-start pipeline, variant comparison, stability test) document v0; they
reference the old folder layout and are not maintained.
