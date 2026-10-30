"""Quality control: scores per registration stage, step-by-step figures, side-by-side comparison of 50 um runs."""
import json

import numpy as np
from scipy import ndimage as ndi

from .atlas import MAJOR, MIDLINE_UM
from .elastix import dice, nmi, outline_mask, load_tp, subset_tp, moving_coords, slab_grid_geom
from .pipeline import FINE_UM, _flip

STAGES = [(1, "start pose"), (2, "similarity"), (3, "constrained affine"), (4, "B-spline 50 um"), (5, "B-spline 20 um")]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def stage_scores(ws, log=print):
    """Score every stage on the same 20 um slab grid against the 10 um template (deterministic): outline Dice
    (natural surface only), NMI of the 540 and 630 backgrounds vs the warped template inside tissue.
    Writes 04_reg_20um/stage_scores.json."""
    d = ws.reg20
    g = json.load(open(d / "geometry.json"))
    sl = _flip(g["flip"])
    org = np.array(g["moving_origin_mm"])
    shape = ws.slab.grid_shape(FINE_UM)
    sp, fo = slab_grid_geom(g, FINE_UM, shape, ws.n50x)
    F, G, mask = (np.load(d / f"slab20_{k}.npy") for k in ("F_540_bg", "G_630_bg", "mask"))
    om = sl(outline_mask(sl(mask), dil=20))
    t10 = ws.atlas.template10()
    tp = load_tp(d)
    rows = []
    for n, name in STAGES:
        ccf = (sl(moving_coords(subset_tp(tp, n), shape, sp, fo)) - org)[..., ::-1] * 100     # 10 um index
        lo = np.maximum(np.floor(ccf[mask].min(0)).astype(int) - 2, 0)
        hi = np.minimum(np.ceil(ccf[mask].max(0)).astype(int) + 3, t10.shape)
        sub = np.asarray(t10[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
        w = ndi.map_coordinates(sub, [ccf[..., k] - lo[k] for k in range(3)], order=1, cval=0)
        del ccf
        thr = 0.05 * np.percentile(w[w > 0], 99)
        m = mask & (w > thr)
        r = dict(stage=name, dice=dice((w > thr) & om, mask), nmi540=nmi(F[m], w[m]), nmi630=nmi(G[m], w[m]))
        rows.append(r)
        log("%-20s outline dice %.3f  nmi540 %.4f  nmi630 %.4f" % (name, r["dice"], r["nmi540"], r["nmi630"]))
    json.dump(rows, open(d / "stage_scores.json", "w"), indent=1)
    return rows


# --------------------------------------------------------------------------------------------------------------

SHOW = ["Isocortex", "HPF", "STR", "TH", "MB", "CB", "fiber tracts"]
PAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
INK, GRID = "#0b0b0b", "#e1e0d9"
C_ACQ, C_FLIP = "#3987e5", "#d95926"


def _sample10(arr, ccf, order):
    """Sample a 10 um atlas volume (memmap) at CCF um positions; 0 outside the volume."""
    i = ccf / 10.0
    outside = np.any([(i[..., k] < 0) | (i[..., k] > arr.shape[k] - 1) for k in range(3)], 0)
    if outside.all():
        return np.zeros(ccf.shape[:2], arr.dtype)
    v = i[~outside]
    lo = np.maximum(np.floor(v.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(v.max(0)).astype(int) + 3, np.array(arr.shape))
    sub = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    c = [np.clip(i[..., k] - lo[k], 0, sub.shape[k] - 1) for k in range(3)]
    if order == 0:
        out = sub[tuple(np.round(ck).astype(int) for ck in c)]
    else:
        out = ndi.map_coordinates(sub.astype(np.float32), c, order=1)
    out[outside] = 0
    return out


def figures(ws, log=print):
    """Step-by-step figures in figures/, original slab orientation on the 20 um grid, labels from the 10 um atlas:
    fig1_preprocessing, fig2_rough_positioning, fig3_steps_overview, fig4_final_detail."""
    plt = _plt()
    from skimage.segmentation import find_boundaries
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white"})
    d20, s, atlas = ws.reg20, ws.slab, ws.atlas
    g = json.load(open(d20 / "geometry.json"))
    flip = g["flip"]
    org = np.array(g["moving_origin_mm"])
    shape = s.grid_shape(FINE_UM)
    nx = shape[2]
    sp, fo = slab_grid_geom(g, FINE_UM, shape, ws.n50x)
    F, mask = np.load(d20 / "slab20_F_540_bg.npy"), np.load(d20 / "slab20_mask.npy")
    t10, a10 = atlas.template10(), atlas.annot10()
    xs = [int(nx * f) for f in (0.27, 0.5, 0.8)]

    def plane_ccf(tp, xo):
        xr = nx - 1 - xo if flip else xo                                   # registration-frame x
        q = moving_coords(tp, (shape[0], shape[1], 1), sp, (fo[0] + xr * sp, fo[1], fo[2]))[:, :, 0]
        return (q - org)[..., ::-1] * 1000

    def outlines(a, lab, m):
        maj = atlas.to_major(lab)
        for col, n in zip(PAL, SHOW):
            b = find_boundaries(maj == list(MAJOR).index(n) + 1, mode="inner") & m
            if b.any():
                a.contour(b.T, levels=[0.5], colors=[col], linewidths=1.0)

    def legend(fig, y=0.005):
        fig.legend([plt.Line2D([], [], color=c, lw=3) for c in PAL], SHOW, loc="lower center", ncol=len(SHOW),
                   frameon=False, bbox_to_anchor=(0.5, y))

    def clean(a, title=None):
        a.set_xticks([])
        a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_visible(False)
        if title:
            a.set_title(title, color=INK)

    # figure 1: preprocessing
    lvl = s.load_prep_level()
    xo = xs[1]
    raw = [s.resample_prep(lvl[c], FINE_UM)[:, :, xo] for c in (s.ch_540, s.ch_630)]
    del lvl
    G = np.load(d20 / "slab20_G_630_bg.npy")
    sl = _flip(flip)
    om = sl(outline_mask(sl(mask), dil=20))
    m = mask[:, :, xo]
    fig, ax = plt.subplots(2, 3, figsize=(24, 9.5))
    for r, (rw, bg, name) in enumerate([(raw[0], F[:, :, xo], "540 nm"), (raw[1], G[:, :, xo], "630 nm")]):
        hi = np.percentile(rw[m], 99.7)
        ax[r, 0].imshow(rw.T, cmap="gray", vmax=hi)
        clean(ax[r, 0], f"{name}: raw (20 um)")
        ax[r, 1].imshow(bg.T, cmap="gray", vmax=1)
        clean(ax[r, 1], f"{name}: bright objects removed (grey opening at {s.prep_um:.0f} um)")
        if r == 0:
            a, b = np.percentile(rw[m], [1, 99])
            rem = np.clip((rw - a) / (b - a) - bg, 0, None) * m
            ax[r, 2].imshow(rem.T, cmap="magma", vmax=np.percentile(rem[m], 99.8))
            clean(ax[r, 2], "540 nm: removed signal (plaques / blobs)")
        else:
            ax[r, 2].imshow(rw.T, cmap="gray", vmax=hi)
            ax[r, 2].contour(m.T, levels=[0.5], colors=[PAL[0]], linewidths=1.2)
            ax[r, 2].contour((om[:, :, xo] & ~m).T, levels=[0.5], colors=[PAL[3]], linewidths=1.0)
            clean(ax[r, 2], "tissue mask (blue); outline-term region outside tissue (yellow)")
    fig.suptitle(f"Step 0 - preprocessing (slab plane x = {xo} of {nx}, 20 um grid)", color=INK, fontsize=14)
    plt.tight_layout(h_pad=2.5)
    plt.savefig(ws.figs / "fig1_preprocessing.png", dpi=70)
    plt.close(fig)
    log("fig1 done")

    # figure 2: rough positioning
    rows = ws.ranked()
    chosen = next(r for r in rows if r["h"] == g["start"])
    fig = plt.figure(figsize=(24, 11))
    gs = fig.add_gridspec(2, 3)
    a = fig.add_subplot(gs[0, 0])
    for f, col, lab in ((0, C_ACQ, "as acquired"), (1, C_FLIP, "flipped")):
        rr = [r for r in rows if r["h"][0] == f]
        a.scatter([r["centre_ml"] for r in rr], [r["score"] for r in rr], s=22, color=col, label=lab,
                  edgecolors="white", linewidths=0.6)
    a.scatter([chosen["centre_ml"]], [chosen["score"]], s=160, facecolors="none", edgecolors=INK, linewidths=1.5,
              label="chosen")
    a.set_xlabel("final slab-centre distance from midline (um)")
    a.set_ylabel("score (z nmi540 + z nmi630 + z outline Dice)")
    a.legend(frameon=False)
    a.set_title(f"{len(rows)} starts: final pose vs score", color=INK)
    a.grid(color=GRID, lw=0.6)
    a = fig.add_subplot(gs[0, 1])
    for f, col in ((0, C_ACQ), (1, C_FLIP)):
        rr = [r for r in rows if r["h"][0] == f]
        a.scatter([r["yaw"] for r in rr], [r["roll"] for r in rr], s=22, color=col, edgecolors="white", linewidths=0.6)
    a.scatter([chosen["yaw"]], [chosen["roll"]], s=160, facecolors="none", edgecolors=INK, linewidths=1.5)
    a.set_xlabel("final yaw, about DV (deg)")
    a.set_ylabel("final roll, about AP (deg)")
    a.set_title("final tilt of the slab plane", color=INK)
    a.grid(color=GRID, lw=0.6)
    a = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    a.scatter([r["h"][1] + rng.uniform(-40, 40) for r in rows], [r["centre_ml"] for r in rows], s=18,
              c=[C_FLIP if r["h"][0] else C_ACQ for r in rows], edgecolors="white", linewidths=0.6)
    a.set_xlabel("start: medial-edge offset from midline (um, jittered)")
    a.set_ylabel("final slab-centre ML (um)")
    a.set_title("start position vs where each run converged", color=INK)
    a.grid(color=GRID, lw=0.6)
    ccf = np.load(d20 / "slab20_to_ccf_um.npy")[mask]
    t50, _ = atlas.load50()
    for k, (pa, proj, lab) in enumerate([(0, t50.max(1), "horizontal (AP x ML)"), (1, t50.max(0), "coronal (DV x ML)")]):
        a = fig.add_subplot(gs[1, k])
        ext = [0, proj.shape[0] * 50, proj.shape[1] * 50, 0]
        a.imshow(proj.T, cmap="gray", extent=ext, aspect="equal")
        H = np.histogram2d(ccf[:, pa], ccf[:, 2], bins=[np.arange(0, ext[1] + 1, 50), np.arange(0, ext[2] + 1, 50)])[0]
        a.contourf(np.arange(25, ext[1], 50), np.arange(25, ext[2], 50), (H > 0).T.astype(float), levels=[0.5, 1.5],
                   colors=[C_FLIP], alpha=0.45)
        a.axhline(MIDLINE_UM, color=C_ACQ, ls="--", lw=1.2)
        a.set_xlabel(("AP" if pa == 0 else "DV") + " (um)")
        a.set_ylabel("ML (um)")
        a.set_title(f"slab tissue in CCF, {lab} projection; dashed = midline", color=INK)
    a = fig.add_subplot(gs[1, 2])
    ml = (ccf[:, 2] - MIDLINE_UM) * np.sign(np.median(ccf[:, 2]) - MIDLINE_UM)
    a.hist(ml, bins=80, color=C_ACQ)
    a.axvline(0, color=INK, lw=1)
    a.set_xlabel("signed distance of slab tissue from midline (um)")
    a.set_ylabel("voxels (20 um)")
    a.set_title("ML extent of the tilted slab (final transform)", color=INK)
    a.grid(color=GRID, lw=0.6, axis="y")
    fig.suptitle("Step 1 - rough positioning: multistart over orientation, ML offset, yaw and roll", color=INK, fontsize=14)
    plt.tight_layout()
    plt.savefig(ws.figs / "fig2_rough_positioning.png", dpi=70)
    plt.close(fig)
    log("fig2 done")

    # figure 3: step overview
    tp_all = load_tp(d20)
    fig, ax = plt.subplots(len(STAGES), 2 * len(xs), figsize=(13 * len(xs), 3.9 * len(STAGES)))
    for r, (n, title) in enumerate(STAGES):
        tp = subset_tp(tp_all, n)
        for c, xo in enumerate(xs):
            cc = plane_ccf(tp, xo)
            lab, tm, m = _sample10(a10, cc, 0), _sample10(t10, cc, 1), mask[:, :, xo]
            a = ax[r, 2 * c]
            a.imshow(F[:, :, xo].T, cmap="gray", vmax=1)
            outlines(a, lab, m)
            clean(a)
            a2 = ax[r, 2 * c + 1]
            a2.imshow(tm.T, cmap="gray", vmax=np.percentile(tm[tm > 0], 99.5) if (tm > 0).any() else 1)
            outlines(a2, lab, m)
            clean(a2)
            if r == 0:
                a.set_title(f"slab 540 background, x={xo}", color=INK)
                a2.set_title(f"atlas template warped to slab, x={xo}", color=INK)
        ax[r, 0].set_ylabel(f"{n}  {title}", color=INK, fontsize=13)
    legend(fig)
    fig.suptitle("Registration steps (rows) at three slab planes; region outlines only inside tissue", color=INK, fontsize=15)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    plt.savefig(ws.figs / "fig3_steps_overview.png", dpi=55)
    plt.close(fig)
    log("fig3 done")

    # figure 4: final detail
    wa, wt = np.load(d20 / "warped_annot20.npy"), np.load(d20 / "warped_template20.npy")
    fig, ax = plt.subplots(len(xs), 2, figsize=(30, 7.4 * len(xs)))
    for r, xo in enumerate(xs):
        m = mask[:, :, xo]
        ax[r, 0].imshow(F[:, :, xo].T, cmap="gray", vmax=1)
        outlines(ax[r, 0], wa[:, :, xo], m)
        clean(ax[r, 0], f"final: slab 540 background + 10 um atlas regions, x={xo}")
        ii, jj = np.indices(m.shape)
        cb = ((ii // 40) + (jj // 40)) % 2 == 0
        ax[r, 1].imshow(np.where(cb, F[:, :, xo], wt[:, :, xo]).T, cmap="gray", vmax=1)
        clean(ax[r, 1], "checkerboard: slab / warped template (800 um squares)")
    legend(fig, 0.0)
    plt.tight_layout(rect=(0, 0.015, 1, 1))
    plt.savefig(ws.figs / "fig4_final_detail.png", dpi=55)
    plt.close(fig)
    log("fig4 done")


def compare_50um(ws, runs=None, log=print):
    """Side-by-side QC of 50 um registrations (dirs relative to the workspace): major regions inside tissue on the
    540 background plus the warped template. Default: best run of each orientation."""
    plt = _plt()
    from skimage.segmentation import find_boundaries
    if runs is None:
        rows = ws.ranked()
        runs = []
        for f, name in ((0, "as acquired"), (1, "flipped")):
            k = next(i + 1 for i, r in enumerate(rows) if r["h"][0] == f)
            runs.append((f"{name} (rank {k})", f"02_pose_search/refine_rank{k}"))
    show = ["HPF", "STR", "TH", "CB", "fiber tracts", "VS", "Isocortex"]
    smask, img, _ = ws.load_prep()
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    nx = smask.shape[2]
    xs = [int(nx * f) for f in (0.2, 0.5, 0.8)]
    fig, ax = plt.subplots(len(xs), 2 * len(runs), figsize=(15 * len(runs), 5.6 * len(xs)))
    for j, (title, d) in enumerate(runs):
        wa, wt = np.load(ws.out / d / "warped_annot.npy"), np.load(ws.out / d / "warped_template.npy")
        maj = ws.atlas.to_major(wa)
        for r, xi in enumerate(xs):
            for c, (im, vmax, lab) in enumerate([(img, 1, "540 background"), (wt, 1.2, "warped template")]):
                a = ax[r, 2 * j + c]
                a.imshow(im[:, :, xi].T, cmap="gray", vmax=vmax)
                for k in [list(MAJOR).index(n) + 1 for n in show]:
                    b = find_boundaries(maj[:, :, xi] == k, mode="inner") & smask[:, :, xi]
                    if b.any():
                        a.contour(b.T, levels=[0.5], colors=[cols[k]], linewidths=0.9)
                a.set_title(f"{title}: {lab}, x={xi}", fontsize=12)
                a.axis("off")
    fig.legend([plt.Line2D([], [], color=cols[list(MAJOR).index(n) + 1], lw=3) for n in show], show,
               loc="lower center", ncol=len(show), fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fn = ws.figs / "qc_compare_orientations.png"
    plt.savefig(fn, dpi=55)
    plt.close(fig)
    log(f"saved {fn}")
    return fn
