"""Use the final registration: TIFF exports in both spaces and Allen regions for cell coordinates."""
import json
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .elastix import load_tp, map_points
from .pipeline import FINE_UM


def export_spaces(ws, log=print):
    """TIFFs with 20 um voxels in 04_reg_20um/export/.
    IMAGE space (slab 20 um grid, original orientation):
        slab_540.tif, slab_630.tif, atlas_template_in_slab.tif, atlas_annotation_in_slab.tif (uint32 Allen ids)
    ATLAS space (full CCFv3 extent at 20 um, axes AP, DV, ML; voxel j centre = 20 j + 5 um):
        slab_540_in_atlas.tif, slab_630_in_atlas.tif, slab_mask_in_atlas.tif, atlas_template20.tif, atlas_annotation20.tif
    The atlas-space volumes need the inverse transform: for each atlas voxel the nearest forward-mapped slab voxel is
    refined with quasi-Newton steps on the forward map (slab20_to_ccf_um.npy). Also writes structures.csv and
    qc_atlas_space.png."""
    import tifffile
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    d, s, um = ws.reg20, ws.slab, FINE_UM
    out = d / "export"
    out.mkdir(exist_ok=True)
    t0 = time.time()

    def save(name, arr, imagej=True):
        kw = dict(resolution=(1 / um, 1 / um), metadata={"spacing": um, "unit": "um", "axes": "ZYX"}) if imagej else \
            dict(resolution=(1e4 / um, 1e4 / um), resolutionunit="CENTIMETER", metadata={"spacing_um": um, "axes": "ZYX"})
        tifffile.imwrite(out / name, arr, imagej=imagej, compression=None if imagej else "zlib", **kw)
        log("  wrote %-32s %s %s" % (name, arr.shape, arr.dtype))

    # image space
    lvl = s.load_prep_level()
    raw = [np.clip(s.resample_prep(lvl[c], um), 0, 65535).astype(np.uint16) for c in (s.ch_540, s.ch_630)]
    del lvl
    mask = np.load(d / "slab20_mask.npy")
    wt, wa = np.load(d / "warped_template20.npy"), np.load(d / "warped_annot20.npy")
    save("slab_540.tif", raw[0])
    save("slab_630.tif", raw[1])
    save("atlas_template_in_slab.tif", np.clip(wt * 20000, 0, 65535).astype(np.uint16))
    save("atlas_annotation_in_slab.tif", wa, imagej=False)          # uint32 ids: plain TIFF (napari labels / Bio-Formats)

    # inverse map: atlas 20 um voxel -> slab 20 um index
    f = np.load(d / "slab20_to_ccf_um.npy")                           # (z,y,x,3) CCF um (AP,DV,ML), original orientation
    dom = ndi.binary_dilation(mask, iterations=3)                      # slab voxels used for the inverse (tissue + rim)
    xs = np.argwhere(dom).astype(np.float32)
    P = f[dom]
    J = np.empty(f.shape[:3] + (3, 3), np.float32)                     # dF/dx: um per slab voxel
    for a in range(3):
        J[..., :, a] = np.gradient(f, axis=a)
    Jinv = np.linalg.inv(J[dom].astype(np.float64)).astype(np.float32)
    del J
    t10, a10 = ws.atlas.template10(), ws.atlas.annot10()
    A20 = tuple(n // 2 for n in t10.shape)
    yidx = lambda p: np.floor((p - 5) / um + 0.5).astype(np.int32)    # CCF um -> atlas 20 um index (centre 20j+5)
    hit = np.zeros(A20, bool)
    ii = np.clip(yidx(P), 0, np.array(A20) - 1)
    hit[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    hit = ndi.binary_dilation(hit, iterations=1)
    Y = np.argwhere(hit)
    yum = Y * um + 5.0
    dist, nn = cKDTree(P).query(yum, distance_upper_bound=1.5 * um * 1.3, workers=-1)
    ok = np.isfinite(dist)
    Y, yum, nn = Y[ok], yum[ok], nn[ok]
    x = xs[nn].copy()
    Ji = Jinv[nn]
    log("inverse: %d atlas voxels near the slab, Newton refinement (%.0fs)" % (len(Y), time.time() - t0))
    fx = lambda x: np.stack([ndi.map_coordinates(f[..., k], x.T, order=1, mode="nearest") for k in range(3)], 1)
    for it in range(4):
        r = yum - fx(x)
        x = x + np.einsum("nij,nj->ni", Ji, r)
        log("  iter %d: residual |F(x)-y| median %.2f um, 99%% %.2f um" % (
            it, np.median(np.linalg.norm(r, axis=1)), np.percentile(np.linalg.norm(r, axis=1), 99)))
    res = np.linalg.norm(yum - fx(x), axis=1)
    inside = np.all((x >= -0.5) & (x <= np.array(mask.shape) - 0.5), 1) & (res < 10)
    log("  final residual median %.2f um, 99%% %.2f um; kept %d voxels (residual < 10 um, inside slab grid)"
        % (np.median(res), np.percentile(res, 99), inside.sum()))
    Y, x = Y[inside], x[inside]

    def to_atlas(vol, order):
        o = np.zeros(A20, vol.dtype)
        o[Y[:, 0], Y[:, 1], Y[:, 2]] = ndi.map_coordinates(vol, x.T, order=order, mode="nearest").astype(vol.dtype)
        return o

    # atlas space
    s540, s630, smk = to_atlas(raw[0], 1), to_atlas(raw[1], 1), to_atlas(mask.astype(np.uint8), 0)
    save("slab_540_in_atlas.tif", s540)
    save("slab_630_in_atlas.tif", s630)
    save("slab_mask_in_atlas.tif", smk * 255)
    t20 = np.asarray(t10, np.float32)[:2 * A20[0], :2 * A20[1], :2 * A20[2]].reshape(
        A20[0], 2, A20[1], 2, A20[2], 2).mean((1, 3, 5)).astype(np.uint16)
    a20 = np.asarray(a10[::2, ::2, ::2])[:A20[0], :A20[1], :A20[2]]   # nearest (voxel 2j, 5 um from the block centre)
    save("atlas_template20.tif", t20)
    save("atlas_annotation20.tif", a20, imagej=False)
    st = ws.atlas.structures
    pd.DataFrame([(k, v["acronym"], v["name"]) for k, v in st.items()], columns=["id", "acronym", "name"]).to_csv(
        out / "structures.csv", index=False)

    # QC: slab warped into atlas space, atlas regions native
    ml_planes = np.percentile(np.argwhere(smk > 0)[:, 2], [20, 50, 80]).astype(int)
    fig, ax = plt.subplots(2, 3, figsize=(30, 11))
    maj = ws.atlas.to_major(a20[:, :, ml_planes])
    for c, m in enumerate(ml_planes):
        sl = s540[:, :, m].astype(np.float32)
        hi = np.percentile(sl[sl > 0], 99.5) if (sl > 0).any() else 1
        b = find_boundaries(maj[:, :, c], mode="inner")
        ax[0, c].imshow(t20[:, :, m].T, cmap="gray")
        ax[0, c].contour(b.T, levels=[0.5], colors=["#3987e5"], linewidths=0.6)
        ax[0, c].set_title(f"atlas template, CCF ML = {m * 20 + 5} um (major regions)")
        ax[1, c].imshow(sl.T, cmap="gray", vmax=hi)
        ax[1, c].contour(b.T, levels=[0.5], colors=["#3987e5"], linewidths=0.6)
        ax[1, c].set_title("slab 540 raw warped into atlas space, same plane")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig(out / "qc_atlas_space.png", dpi=55)
    plt.close(fig)
    log("export done %.0fs" % (time.time() - t0))


def _lookup(vol, idx):
    i = np.round(idx).astype(int)
    ok = np.all((i >= 0) & (i < np.array(vol.shape)), 1)
    out = np.zeros(len(i), vol.dtype)
    out[ok] = vol[i[ok, 0], i[ok, 1], i[ok, 2]]
    return out


def map_cells(ws, pts, regdir=None):
    """Allen regions and CCF coordinates of cells at level-0 voxel positions pts (N, 3; z, y, x). Two routes:
      region_id                  : lookup in the 10 um annotation warped onto the 20 um slab grid
      ccf_*_um, region_id_ccf    : transformix on the points (slab -> CCF, honours the flip), then the 10 um annotation
    Returns a DataFrame."""
    regdir = regdir or ws.reg20
    pts = np.asarray(pts, float).reshape(-1, 3)
    g = json.load(open(regdir / "geometry.json"))
    org = np.array(g["moving_origin_mm"])
    st = ws.atlas.structures
    ids = _lookup(np.load(regdir / "warped_annot20.npy"), ws.slab.l0_to_grid(pts, FINE_UM))
    i50 = ws.slab.l0_to_grid(pts, 50.0)                               # fixed physical point in the registration frame
    if g["flip"]:
        i50[:, 2] = (ws.n50x - 1) - i50[:, 2]
    ccf = (map_points(load_tp(regdir), i50[:, ::-1] * g["fixed_spacing_mm"]) - org)[:, ::-1] * 1000   # CCF um
    a10 = ws.atlas.annot10()
    ids_ccf = np.array([a10[tuple(i)] if np.all((i >= 0) & (i < a10.shape)) else 0
                        for i in np.round(ccf / 10).astype(int)])
    acr = lambda v: [st.get(int(i), {}).get("acronym", "outside") for i in v]
    return pd.DataFrame(dict(z=pts[:, 0], y=pts[:, 1], x=pts[:, 2], region_id=ids, acronym=acr(ids),
                             ccf_ap_um=ccf[:, 0], ccf_dv_um=ccf[:, 1], ccf_ml_um=ccf[:, 2],
                             region_id_ccf=ids_ccf, acronym_ccf=acr(ids_ccf)))


def self_test(ws, log=print):
    """One synthetic point per major region (region voxel nearest its centroid): the region must be found back
    and both routes of map_cells must agree."""
    from .atlas import MAJOR
    st = ws.atlas.structures
    wa = np.load(ws.reg20 / "warped_annot20.npy")
    maj = ws.atlas.to_major(wa)
    mask = np.load(ws.reg20 / "slab20_mask.npy")
    rows = []
    for k, name in enumerate(MAJOR):
        vox = np.argwhere((maj == k + 1) & mask)
        if len(vox) < 200:
            continue
        c = vox[np.argmin(((vox - vox.mean(0)) ** 2).sum(1))]
        rows.append((name, ws.slab.grid_to_l0(c, FINE_UM)))
    res = map_cells(ws, np.array([r[1] for r in rows]))
    res.insert(0, "expected", [r[0] for r in rows])
    res["major_ok"] = [MAJOR[e] in st.get(int(i), {}).get("parents", []) for e, i in zip(res.expected, res.region_id)]
    res["routes_agree"] = res.region_id == res.region_id_ccf
    log(res.round(0).to_string(index=False))
    log("major region correct: %d/%d   lookup and transformix routes agree: %d/%d" % (
        res.major_ok.sum(), len(res), res.routes_agree.sum(), len(res)))
    return res
