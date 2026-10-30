"""Registration of a sagittal slab to Allen CCFv3, step by step (see README for the reasons behind each step).

    ws = Workspace(zarr)                 # results in <zarr name>_atlas/ next to the zarr
    prepare(ws)                          # 50 um registration images + tissue mask            ~10 s
    pose_search(ws)                      # multistart rigid/similarity/affine                 ~10 min
    rank_poses(ws)                       # final pose + deterministic score per run           ~20 s
    orientation_test(ws)                 # which face is medial                               ~6 min
    choose_pose(ws)                      # best run of the chosen orientation -> 03_reg_50um   (B-spline at 50 um)
    refine_20um(ws)                      # B-spline at 20 um vs 10 um atlas, 10 um labels     ~2 min

or `run_all(ws)`. Every function takes `log` (a print-like callable) for progress messages.
"""
import itertools
import json
import os
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .atlas import Atlas, ATLAS_DIR, MIDLINE_UM
from .slab import Slab, tissue_mask
from .elastix import (ATLAS_SP, to_itk, atlas_mask, prep_atlas, dice, nmi, param_map, multi_metric, run_elastix,
                      write_tp, clean_intermediates, load_tp, po_from_dicts, tp_to_dicts, moving_coords, map_points,
                      outline_mask, slab_grid_geom)

ATLAS_UM = 50.0
FINE_UM = 20.0
SCALE0 = 1.15                         # start CCF/slab size ratio (the similarity stage finds the final value)
OFFS_UM = [0, 250, 500, 750, 1000]    # start: medial face distance from the midline
ROLL = [-20, -10, 0, 10, 20]
YAW = [-30, -20, -10, 0, 10, 20, 30]


class Workspace:
    """Paths and settings of one registration. Settings are stored in <out>/workspace.json and reused when the
    workspace is opened again with only the zarr path."""

    def __init__(self, zarr_path, out=None, ch_540=None, ch_630=None, atlas_dir=None, cache_dir=None):
        zarr_path = Path(zarr_path)
        self.out = Path(out) if out else zarr_path.with_name(zarr_path.name.removesuffix(".zarr") + "_atlas")
        cfg_fn = self.out / "workspace.json"
        cfg = json.load(open(cfg_fn)) if cfg_fn.exists() else {}
        given = dict(ch_540=ch_540, ch_630=ch_630, atlas_dir=atlas_dir and str(atlas_dir),
                     cache_dir=cache_dir and str(cache_dir))
        defaults = dict(ch_540=0, ch_630=1, atlas_dir=None, cache_dir=None)
        self.cfg = {k: given[k] if given[k] is not None else cfg.get(k, defaults[k]) for k in given}
        self.cfg["zarr"] = str(zarr_path)
        self.slab = Slab(zarr_path, self.cfg["ch_540"], self.cfg["ch_630"])
        self.atlas = Atlas(self.cfg["atlas_dir"] or ATLAS_DIR, cache_dir=self.cfg["cache_dir"])
        self.prep = self.out / "01_prep"
        self.search = self.out / "02_pose_search"
        self.reg50 = self.out / "03_reg_50um"
        self.reg20 = self.out / "04_reg_20um"
        self.figs = self.out / "figures"
        for d in (self.out, self.prep, self.search, self.reg50, self.reg20, self.figs):
            d.mkdir(parents=True, exist_ok=True)
        cfg_fn.write_text(json.dumps(self.cfg, indent=1))

    @property
    def n50x(self):
        return self.slab.grid_shape(ATLAS_UM)[2]

    def status(self):
        """Which steps have results: prepared, searched, ranked, orientation, reg50, reg20."""
        return dict(prepared=(self.prep / "slab50_mask.npy").exists(),
                    searched=(self.search / "multistart.json").exists(),
                    ranked=(self.search / "multistart_ranked.json").exists(),
                    orientation=(self.search / "orientation_test.json").exists(),
                    reg50=(self.reg50 / "warped_annot.npy").exists(),
                    reg20=(self.reg20 / "warped_annot20.npy").exists())

    def geometry(self, fine=True):
        d = self.reg20 if fine and (self.reg20 / "geometry.json").exists() else self.reg50
        return json.load(open(d / "geometry.json"))

    def load_prep(self):
        return (np.load(self.prep / "slab50_mask.npy"), np.load(self.prep / "slab50_F_540_bg.npy"),
                np.load(self.prep / "slab50_G_630_bg.npy"))

    def ranked(self):
        return json.load(open(self.search / "multistart_ranked.json"))

    def orientation(self):
        fn = self.search / "orientation_test.json"
        return json.load(open(fn)) if fn.exists() else None

    def labels(self):
        """Best available warped annotation in image space: (uint32 Allen ids, grid voxel size in um)."""
        if (self.reg20 / "warped_annot20.npy").exists():
            return np.load(self.reg20 / "warped_annot20.npy"), FINE_UM
        if (self.reg50 / "warped_annot.npy").exists():
            return np.load(self.reg50 / "warped_annot.npy"), ATLAS_UM
        return None, None


def _flip(flip):
    return (lambda a: a[:, :, ::-1]) if flip else (lambda a: a)


# --------------------------------------------------------------------------------------------------------------
# 1. preprocessing

def prepare(ws, log=print):
    """Load the slab at ~16 um, remove bright objects (540 plaques, 630 microglia) at that resolution, resample to
    the 50 um atlas grid, build the tissue mask. Writes 01_prep/: slab50_raw.npy (2,z,y,x), slab50_mask.npy,
    slab50_F_540_bg.npy, slab50_G_630_bg.npy, qc_prepare.png."""
    s = ws.slab
    lvl = s.load_prep_level()
    log(f"prep level {s.prep_level} ({s.prep_um:.1f} um), shape {lvl.shape}")
    raw = np.stack([s.resample_prep(lvl[c], ATLAS_UM) for c in (s.ch_540, s.ch_630)])
    valid = ndi.binary_erosion(s.resample_prep((lvl[s.ch_630] > 0).astype(np.float32), ATLAS_UM) > 0.99, iterations=2)
    mask = tissue_mask(raw[1], valid)
    log(f"slab @50um {raw.shape[1:]}, tissue fraction {mask.mean():.3f}")
    np.save(ws.prep / "slab50_raw.npy", raw)
    np.save(ws.prep / "slab50_mask.npy", mask)
    F, G = s.registration_images(lvl, mask, ATLAS_UM)
    np.save(ws.prep / "slab50_F_540_bg.npy", F.astype(np.float32))
    np.save(ws.prep / "slab50_G_630_bg.npy", G.astype(np.float32))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx = mask.shape[2]
    xs = [int(nx * f) for f in (0.27, 0.5, 0.73)]
    cols = [("raw 540", raw[0]), ("raw 630", raw[1]), ("540 background", F), ("630 background", G)]
    fig, ax = plt.subplots(len(xs), len(cols) + 1, figsize=(3.6 * (len(cols) + 1), 6.5))
    for r, xi in enumerate(xs):
        ax[r, 0].imshow(mask[:, :, xi].T, cmap="gray")
        ax[r, 0].set_title(f"mask x={xi}", fontsize=8)
        for j, (k, im) in enumerate(cols):
            ax[r, j + 1].imshow(im[:, :, xi].T, cmap="gray", vmax=np.percentile(im[mask], 99.5))
            ax[r, j + 1].set_title(k, fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(ws.prep / "qc_prepare.png", dpi=70)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# 2. pose search (worker processes)

_W = {}


def _init_worker(prep_dir, atlas_dir, threads):
    import itk
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
    t, _ = Atlas(atlas_dir).load50()
    _W.update(atl=prep_atlas(t), amask=atlas_mask(t), smask=np.load(Path(prep_dir) / "slab50_mask.npy"),
              F=np.load(Path(prep_dir) / "slab50_F_540_bg.npy"), Gc=np.load(Path(prep_dir) / "slab50_G_630_bg.npy"),
              threads=threads)


def _bbox_centre(m, ax):
    i = np.flatnonzero(m.any(axis=tuple(a for a in range(3) if a != ax)))
    return (i[0] + i[-1]) / 2


def _euler_init_map(po, centre, angles_deg, shape_zyx, fsp):
    m = po.GetDefaultParameterMap("rigid")
    vals = dict(Transform="EulerTransform", NumberOfParameters="6",
                TransformParameters=[str(np.radians(a)) for a in angles_deg] + ["0", "0", "0"],
                CenterOfRotationPoint=[str(c) for c in centre], InitialTransformParameterFileName="NoInitialTransform",
                HowToCombineTransforms="Compose", FixedImageDimension="3", MovingImageDimension="3",
                FixedInternalImagePixelType="float", MovingInternalImagePixelType="float",
                Size=[str(s) for s in shape_zyx[::-1]], Index=["0"] * 3, Spacing=[str(fsp)] * 3, Origin=["0"] * 3,
                Direction=["1", "0", "0", "0", "1", "0", "0", "0", "1"], UseDirectionCosines="true",
                ComputeZYX="false", ResampleInterpolator="FinalLinearInterpolator", Resampler="DefaultResampler",
                DefaultPixelValue="0", ResultImageFormat="nii", ResultImagePixelType="float")
    for k, v in vals.items():
        m[k] = v if isinstance(v, list) else [v]
    return m


def _run_one(h):
    """One start: Euler start pose -> similarity -> constrained affine, scored on all voxels."""
    import itk
    flip, off_um, yaw, roll = h
    sl = _flip(flip)
    smask, F, Gc = sl(_W["smask"]), sl(_W["F"]), sl(_W["Gc"])
    atl, amask = _W["atl"], _W["amask"]
    fsp = ATLAS_SP * SCALE0
    area = smask.sum((0, 1))
    x_med = int(np.flatnonzero(area > 0.05 * area.max())[0])
    org = (x_med * fsp - (MIDLINE_UM + off_um) / 1000.0,
           _bbox_centre(smask, 1) * fsp - _bbox_centre(amask, 1) * ATLAS_SP,
           _bbox_centre(smask, 0) * fsp - _bbox_centre(amask, 0) * ATLAS_SP)
    cz, cy, cx = np.argwhere(smask).mean(0)
    centre = (cx * fsp, cy * fsp, cz * fsp)

    fimg = lambda a, dt=np.float32: to_itk(a, dtype=dt, spacing=fsp)
    FI, GI, FM = fimg(F), fimg(Gc), fimg(smask, np.uint8)
    OMa = outline_mask(smask)
    O, OM = fimg(ndi.gaussian_filter(smask.astype(np.float32), 1.5)), fimg(OMa, np.uint8)
    M, MM = to_itk(atl, org), to_itk(amask, org, np.uint8)
    MO = to_itk(ndi.gaussian_filter(amask.astype(np.float32), 1.5), org)

    po = itk.ParameterObject.New()
    common = multi_metric(3, (0.4, 0.25, 0.35), NumberOfResolutions=3, MaximumNumberOfIterations=250,
                          NumberOfSpatialSamples=3000, MaximumNumberOfThreads=_W["threads"])
    po.AddParameterMap(param_map(po, "rigid", Transform="SimilarityTransform", AutomaticScalesEstimation="true", **common))
    # AffineDTI with the thickness scale and all shears frozen: a free affine collapses the thin slab
    po.AddParameterMap(param_map(po, "affine", Transform="AffineDTITransform", AutomaticScalesEstimation="false",
                                 Scales=[40, 40, 40, 1e10, 1e10, 1e10, 1e10, 40, 40, 1, 1, 1], **common))
    init = itk.ParameterObject.New()
    init.AddParameterMap(_euler_init_map(po, centre, (0, yaw, roll), smask.shape, fsp))
    t0 = time.time()
    try:
        res, tp = run_elastix(po, [(FI, M, FM, MM), (GI, M, FM), (O, MO, OM)], init=init)
    except RuntimeError:
        return dict(h=h, ok=False)
    m = smask & (res > 0.05)
    return dict(h=h, ok=True, t=time.time() - t0, dice=float(dice((res > 0.05) & OMa, smask)),
                nmi540=float(nmi(F[m], res[m])) if m.sum() > 1000 else 0.0,
                nmi630=float(nmi(Gc[m], res[m])) if m.sum() > 1000 else 0.0,
                org=org, fsp=fsp, x_med=x_med, maps=tp_to_dicts(tp))


def pose_search(ws, workers=5, threads=None, test=False, log=print):
    """350 registrations from different start poses: flip (which face is medial) x medial-face offset 0..1000 um
    x start yaw -30..30 deg x start roll -20..20 deg. Writes 02_pose_search/multistart.json."""
    hyps = list(itertools.product([0, 1], OFFS_UM, YAW, ROLL))
    if test:
        hyps, workers = [(0, 0, 0, 0), (0, 500, 10, -10), (1, 500, 0, 0)], 3
    threads = threads or max(1, (os.cpu_count() or 4) // workers)
    log(f"pose search: {len(hyps)} starts, {workers} workers x {threads} threads")
    t0 = time.time()
    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(ws.prep), str(ws.atlas.dir), threads)) as p:
        for i, r in enumerate(p.imap_unordered(_run_one, hyps)):
            rows.append(r)
            if r["ok"] and (i % 10 == 0 or test):
                log("%3d/%d  %.0fs  h=%s dice %.3f nmi540 %.4f nmi630 %.4f" % (
                    i + 1, len(hyps), time.time() - t0, r["h"], r["dice"], r["nmi540"], r["nmi630"]))
    json.dump(rows, open(ws.search / ("multistart_test.json" if test else "multistart.json"), "w"))
    log("pose search done %.0fs, failed %d" % (time.time() - t0, sum(not r["ok"] for r in rows)))
    return rows


# --------------------------------------------------------------------------------------------------------------
# 3. ranking

def rank_poses(ws, fn="multistart.json", log=print):
    """Final pose of each run in CCF (medial-face ML, yaw/roll vs sagittal, scale) and the deterministic score
    z(nmi540) + z(nmi630) + z(dice). Writes multistart_ranked.json and multistart_summary.png."""
    fn = ws.search / fn
    rows = [r for r in json.load(open(fn)) if r["ok"]]
    smask0 = np.load(ws.prep / "slab50_mask.npy")
    mid = MIDLINE_UM / 1000
    for r in rows:
        flip = r["h"][0]
        sm = smask0[:, :, ::-1] if flip else smask0
        f = r["fsp"]
        vox = np.argwhere(sm)
        c = vox.mean(0)
        face = vox[vox[:, 2] <= r["x_med"] + 2].mean(0)
        pts = np.array([c, c + [0, 0, 1 / f], c + [0, 1 / f, 0], c + [1 / f, 0, 0], face])[:, ::-1] * f   # 1 mm steps
        q = map_points(po_from_dicts(r["maps"]), pts) - np.array(r["org"])                       # CCF mm (ML,DV,AP)
        ax_x, ax_y, ax_z = q[1] - q[0], q[2] - q[0], q[3] - q[0]
        n = ax_x / np.linalg.norm(ax_x)
        side = np.sign(q[0][0] - mid) or 1
        r.update(face_ml=float((q[4][0] - mid) * side * 1000), centre_ml=float((q[0][0] - mid) * side * 1000),
                 yaw=float(np.degrees(np.arctan2(n[2], abs(n[0])))), roll=float(np.degrees(np.arctan2(n[1], abs(n[0])))),
                 thick_scale=float(np.linalg.norm(ax_x)),
                 inplane_scale=float((np.linalg.norm(ax_y) + np.linalg.norm(ax_z)) / 2))
    S = lambda k: np.array([r[k] for r in rows])
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    score = z(S("nmi540")) + z(S("nmi630")) + z(S("dice"))
    for r, s in zip(rows, score):
        r["score"] = float(s)
    rows.sort(key=lambda r: -r["score"])
    log("%d runs. score = z(nmi540)+z(nmi630)+z(dice)" % len(rows))
    log(" rank  flip off yaw0 roll0 | score  nmi540  nmi630  dice | face_ml centre_ml  yaw   roll  scale_in scale_thick")
    for i, r in enumerate(rows[:10]):
        log("%5s  %d  %4d %4d %4d | %5.2f %.4f %.4f %.3f | %6.0f %7.0f %6.1f %6.1f  %.2f  %.2f" % (
            i + 1, *r["h"], r["score"], r["nmi540"], r["nmi630"], r["dice"],
            r["face_ml"], r["centre_ml"], r["yaw"], r["roll"], r["inplane_scale"], r["thick_scale"]))
    top = rows[:10]
    log("top-10 pose spread: face_ml %.0f +- %.0f um, yaw %.1f +- %.1f, roll %.1f +- %.1f" % (
        np.mean([r["face_ml"] for r in top]), np.std([r["face_ml"] for r in top]), np.mean([r["yaw"] for r in top]),
        np.std([r["yaw"] for r in top]), np.mean([r["roll"] for r in top]), np.std([r["roll"] for r in top])))
    json.dump(rows, open(fn.with_name(fn.stem + "_ranked.json"), "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    c = ["C0" if r["h"][0] == 0 else "C3" for r in rows]
    for a, (k, lab) in zip(ax, [("face_ml", "final medial-face distance to midline (um)"), ("yaw", "final yaw (deg)"),
                                ("roll", "final roll (deg)")]):
        a.scatter(S(k), S("score"), c=c, s=12)
        a.set_xlabel(lab)
        a.set_ylabel("score")
    ax[3].scatter([r["h"][1] for r in rows], S("face_ml"), c=c, s=12)
    ax[3].plot([0, 1000], [0, 1000], "k:")
    ax[3].set_xlabel("start medial-face offset (um)")
    ax[3].set_ylabel("final (um)")
    ax[0].set_title("blue = slab as acquired, red = flipped (other face medial)")
    plt.tight_layout()
    plt.savefig(ws.search / "multistart_summary.png", dpi=60)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------------------------------------------
# 4. B-spline at 50 um

def refine_50um(ws, rank, log=print):
    """B-spline refinement (final control-point spacing 0.6 mm) of ranked run `rank` (1-based) at 50 um.
    Writes 02_pose_search/refine_rank<rank>/: TransformParameters.*.txt, geometry.json (incl. flip),
    warped_template.npy and warped_annot.npy in the ORIGINAL slab orientation. Returns that folder."""
    import itk
    from brainglobe_registration.elastix.register import convert_atlas_labels, restore_atlas_labels
    best = ws.ranked()[rank - 1]
    flip, org, fsp = best["h"][0], tuple(best["org"]), best["fsp"]
    out = ws.search / f"refine_rank{rank}"
    out.mkdir(exist_ok=True)
    sl = _flip(flip)
    smask, F, Gc = (sl(a) for a in ws.load_prep())
    t, annot = ws.atlas.load50()
    atl, amask = prep_atlas(t), atlas_mask(t)

    fimg = lambda a, dt=np.float32: to_itk(a, dtype=dt, spacing=fsp)
    FI, GI, FM = fimg(F), fimg(Gc), fimg(smask, np.uint8)
    OMa = outline_mask(smask)
    O, OM = fimg(ndi.gaussian_filter(smask.astype(np.float32), 1.5)), fimg(OMa, np.uint8)
    M, MM = to_itk(atl, org), to_itk(amask, org, np.uint8)
    MO = to_itk(ndi.gaussian_filter(amask.astype(np.float32), 1.5), org)

    po = itk.ParameterObject.New()
    po.AddParameterMap(param_map(po, "bspline", **multi_metric(
        4, (0.35, 0.2, 0.25, 0.2), NumberOfResolutions=3, FinalGridSpacingInPhysicalUnits=0.6,
        GridSpacingSchedule=[4, 2, 1], NumberOfSpatialSamples=4096, MaximumNumberOfIterations=500)))
    t0 = time.time()
    res, tp = run_elastix(po, [(FI, M, FM, MM), (GI, M, FM), (O, MO, OM), (FI, M, FM)],
                          init=po_from_dicts(best["maps"]), log_dir=out)
    log("rank %d: bspline 50 um %.1fs" % (rank, time.time() - t0))
    write_tp(tp, out)
    lab16, mapping = convert_atlas_labels(annot.astype(np.uint32))
    for i in range(tp.GetNumberOfParameterMaps()):
        tp.SetParameter(i, "ResampleInterpolator", "FinalNearestNeighborInterpolator")
    wa = restore_atlas_labels(np.asarray(itk.transformix_filter(to_itk(lab16, org), tp)).round().astype(np.uint16), mapping)
    m = smask & (res > 0.05)
    log("rank %d: dice %.3f nmi540 %.4f nmi630 %.4f (linear: dice %.3f nmi540 %.4f nmi630 %.4f)" % (
        rank, dice((res > 0.05) & OMa, smask), nmi(F[m], res[m]), nmi(Gc[m], res[m]),
        best["dice"], best["nmi540"], best["nmi630"]))
    np.save(out / "warped_template.npy", sl(res).astype(np.float32))
    np.save(out / "warped_annot.npy", sl(wa))
    json.dump(dict(variant="F_540_bg+G_630_bg", flip=int(flip), fixed_spacing_mm=fsp, moving_spacing_mm=ATLAS_SP,
                   moving_origin_mm=org, x_medial=best["x_med"], start=best["h"], rank=rank,
                   pose=dict(face_ml_um=best["face_ml"], yaw=best["yaw"], roll=best["roll"]),
                   note="transform defined on the slab flipped along x if flip=1 (x' = nx-1-x); "
                        "saved npy outputs are in original orientation"),
              open(out / "geometry.json", "w"), indent=1)
    clean_intermediates(out)
    return out


def _refined(ws, rank, log):
    d = ws.search / f"refine_rank{rank}"
    if not (d / "warped_template.npy").exists():
        refine_50um(ws, rank, log=log)
    return d


# --------------------------------------------------------------------------------------------------------------
# 5. orientation

def orientation_test(ws, n=5, log=print):
    """B-spline-refine the top n runs of EACH orientation and compare deterministic final scores. Decide only if
    every score whose gap exceeds twice the within-orientation spread favours the same orientation (ties ignored).
    Writes orientation_test.json; returns the verdict: 'flipped', 'as acquired' or 'undecided'."""
    rows = ws.ranked()
    ranks = {f: [i + 1 for i, r in enumerate(rows) if r["h"][0] == f][:n] for f in (0, 1)}
    sm0, F0, G0 = ws.load_prep()
    res = {0: [], 1: []}
    for f, rk in ranks.items():
        for k in rk:
            w = np.load(_refined(ws, k, log) / "warped_template.npy")      # saved in original orientation
            sm, F, G = (a[:, :, ::-1] for a in (sm0, F0, G0)) if f else (sm0, F0, G0)
            w = w[:, :, ::-1] if f else w                                   # back to the frame of the outline mask
            m = sm & (w > 0.05)
            res[f].append((dice((w > 0.05) & outline_mask(sm), sm), nmi(F[m], w[m]), nmi(G[m], w[m])))
            log("flip=%d rank %3d  dice %.4f  nmi540 %.4f  nmi630 %.4f" % (f, k, *res[f][-1]))
    wins = []
    for j, name in enumerate(["dice", "nmi540", "nmi630"]):
        a = np.array([r[j] for r in res[0]])
        b = np.array([r[j] for r in res[1]])
        gap = b.mean() - a.mean()
        spread = max(a.std(), b.std(), 1e-9)
        wins.append(np.sign(gap) if abs(gap) > 2 * spread else 0)
        log("%-7s as-acquired %.4f +- %.4f | flipped %.4f +- %.4f | gap/spread %+.1f" % (
            name, a.mean(), a.std(), b.mean(), b.std(), gap / spread))
    # after the B-spline the outline is fitted in both orientations, so Dice carries little orientation
    # information; decide on the significant scores, provided none significantly contradicts
    sig = [w for w in wins if w != 0]
    verdict = ("flipped" if sig[0] > 0 else "as acquired") if sig and len(set(sig)) == 1 else "undecided"
    log(f"orientation verdict: {verdict}")
    json.dump(dict(ranks=ranks, scores={str(k): v for k, v in res.items()}, verdict=verdict),
              open(ws.search / "orientation_test.json", "w"), indent=1)
    return verdict


def choose_pose(ws, flip=None, log=print):
    """Copy the B-spline result of the best-ranked run with orientation `flip` (0 = as acquired, 1 = flipped;
    None = verdict of the orientation test) to 03_reg_50um/. Returns the chosen rank."""
    if flip is None:
        o = ws.orientation()
        if o is None or o["verdict"] == "undecided":
            raise ValueError("orientation undecided: compare the two orientations and pass flip=0 or flip=1")
        flip = int(o["verdict"] == "flipped")
    rank = next(i + 1 for i, r in enumerate(ws.ranked()) if r["h"][0] == flip)
    src = _refined(ws, rank, log)
    shutil.rmtree(ws.reg50, ignore_errors=True)
    shutil.copytree(src, ws.reg50)
    log(f"50 um registration: rank {rank} ({'flipped' if flip else 'as acquired'}) -> {ws.reg50}")
    return rank


# --------------------------------------------------------------------------------------------------------------
# 6. fine B-spline at 20 um vs the 10 um atlas

def refine_20um(ws, post_only=False, log=print):
    """Fine B-spline (final spacing 0.25 mm) at 20 um against the 10 um template (2x block-mean, cropped around the
    slab + 1 mm), initialised with 03_reg_50um. Labels come from the 10 um annotation via the deformation field
    (nearest neighbour). Writes 04_reg_20um/: TransformParameters.*.txt, geometry.json, slab20_*.npy,
    warped_template20.npy, warped_annot20.npy (uint32, 20 um slab grid, original orientation), slab20_to_ccf_um.npy.
    post_only: reuse the saved transforms, redo labels/template only."""
    import itk
    um = FINE_UM
    src, out, s = ws.reg50, ws.reg20, ws.slab
    g = json.load(open(src / "geometry.json"))
    flip = g["flip"]
    org = np.array(g["moving_origin_mm"])
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(os.cpu_count() or 4)
    t0 = time.time()

    # fixed: slab at 20 um (same preprocessing as 50 um)
    shape = s.grid_shape(um)
    sm50 = np.load(ws.prep / "slab50_mask.npy")
    mask = s.resample_grid(sm50, 50, um) > 0.5
    F, G = s.registration_images(s.load_prep_level(), mask, um)
    np.save(out / "slab20_mask.npy", mask)
    np.save(out / "slab20_F_540_bg.npy", F)
    np.save(out / "slab20_G_630_bg.npy", G)
    sl = _flip(flip)
    Fr, Gr, mr = sl(F), sl(G), sl(mask)
    sp, fo = slab_grid_geom(g, um, shape, ws.n50x)
    log("fixed 20 um %s spacing %.5f origin %s (%.0fs)" % (shape, sp, np.round(fo, 4), time.time() - t0))

    def fimg(a, dt=np.float32):
        im = to_itk(a, dtype=dt, spacing=sp)
        im.SetOrigin(fo)
        return im
    FI, GI, FM = fimg(Fr), fimg(Gr), fimg(mr, np.uint8)
    OMa = outline_mask(mr, dil=20)
    O, OM = fimg(ndi.gaussian_filter(mr.astype(np.float32), 3.0)), fimg(OMa, np.uint8)

    # moving: 10 um template -> 20 um block mean, cropped around the slab footprint (+1 mm)
    t10 = ws.atlas.template10()
    ccf50 = moving_coords(load_tp(src), sm50.shape, g["fixed_spacing_mm"])      # registration frame
    ccf50 = (sl(ccf50) - org)[..., ::-1] * 1000                                  # CCF um (AP,DV,ML), original orientation
    lo = np.maximum(np.floor((ccf50[sm50].min(0) - 1000) / 20).astype(int), 0)
    hi = np.minimum(np.ceil((ccf50[sm50].max(0) + 1000) / 20).astype(int), np.array(t10.shape) // 2)
    crop = np.asarray(t10[2 * lo[0]:2 * hi[0], 2 * lo[1]:2 * hi[1], 2 * lo[2]:2 * hi[2]], np.float32)
    t20 = crop.reshape(hi[0] - lo[0], 2, hi[1] - lo[1], 2, hi[2] - lo[2], 2).mean((1, 3, 5))
    del crop
    morg = tuple(org + 0.005 + 0.020 * lo[::-1])                                 # itk order; block centre = 20j + 5 um
    mimg = lambda a, dt=np.float32: to_itk(a, morg, dt, spacing=0.02)
    am20 = atlas_mask(t20)
    M, MM = mimg(prep_atlas(t20, sigma=1.0)), mimg(am20, np.uint8)
    MO = mimg(ndi.gaussian_filter(am20.astype(np.float32), 3.0))
    log("moving 20 um crop %s, CCF um from %s (%.0fs)" % (t20.shape, lo * 20, time.time() - t0))

    if not post_only:
        po = itk.ParameterObject.New()
        po.AddParameterMap(param_map(po, "bspline", **multi_metric(
            4, (0.4, 0.2, 0.2, 0.2), NumberOfResolutions=3, FinalGridSpacingInPhysicalUnits=0.25,
            GridSpacingSchedule=[4, 2, 1], NumberOfSpatialSamples=8192, MaximumNumberOfIterations=800,
            NumberOfHistogramBins=48)))
        t1 = time.time()
        _, tp = run_elastix(po, [(FI, M, FM, MM), (GI, M, FM), (O, MO, OM), (FI, M, FM)], init=load_tp(src), log_dir=out)
        log("fine bspline %.0fs" % (time.time() - t1))
        write_tp(tp, out)
        g20 = dict(g, fine_um=um, fine_spacing_mm=sp, fine_origin_mm=fo, fine_shape=shape, init_from="03_reg_50um")
        json.dump(g20, open(out / "geometry.json", "w"), indent=1)

    # labels from the 10 um annotation via the deformation field
    q = moving_coords(load_tp(out), shape, sp, fo)                                # registration frame
    ccf = (sl(q) - org)[..., ::-1] * 1000                                         # CCF um, original orientation
    del q
    np.save(out / "slab20_to_ccf_um.npy", ccf.astype(np.float32))
    a10 = ws.atlas.annot10()
    i10 = np.round(ccf / 10).astype(np.int32)
    inside = np.all((i10 >= 0) & (i10 < np.array(a10.shape)), -1)
    l0, l1 = i10[inside].min(0), i10[inside].max(0) + 1
    asub = np.asarray(a10[l0[0]:l1[0], l0[1]:l1[1], l0[2]:l1[2]])
    wa = np.zeros(shape, np.uint32)
    j = i10[inside] - l0
    wa[inside] = asub[j[:, 0], j[:, 1], j[:, 2]]
    np.save(out / "warped_annot20.npy", wa)
    j20 = (ccf - 5) / 20 - lo                                                     # 20 um crop index (centres 20j+5)
    wt = ndi.map_coordinates(t20, [j20[..., 0], j20[..., 1], j20[..., 2]], order=1, cval=0).astype(np.float32)
    np.save(out / "warped_template20.npy", wt / np.percentile(wt[wt > 0], 99))
    m = mask & (wt > 0)
    log("20 um: outline dice %.3f  nmi540 %.4f  nmi630 %.4f   total %.0fs" % (
        dice((wt > 0.05 * wt.max()) & sl(OMa), mask), nmi(F[m], wt[m]), nmi(G[m], wt[m]), time.time() - t0))
    clean_intermediates(out)


# --------------------------------------------------------------------------------------------------------------

def find_pose(ws, workers=5, flip=None, log=print):
    """Steps 1-5: prepare, pose search, ranking, orientation test, 50 um result. Skips steps that have results.
    Returns the orientation verdict; with an undecided verdict and flip=None no 50 um result is chosen."""
    st = ws.status()
    if not st["prepared"]:
        prepare(ws, log=log)
    if not st["searched"]:
        pose_search(ws, workers=workers, log=log)
    if not st["ranked"]:
        rank_poses(ws, log=log)
    verdict = ws.orientation()["verdict"] if st["orientation"] else orientation_test(ws, log=log)
    if flip is not None or verdict != "undecided":
        choose_pose(ws, flip, log=log)
    return verdict


def run_all(ws, workers=5, flip=None, log=print):
    verdict = find_pose(ws, workers=workers, flip=flip, log=log)
    if flip is None and verdict == "undecided":
        log("orientation undecided: check both, then run choose_pose(ws, flip) and refine_20um(ws)")
        return verdict
    refine_20um(ws, log=log)
    return verdict
