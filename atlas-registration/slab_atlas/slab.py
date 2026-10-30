"""The slab image (fused OME-Zarr) and its preprocessing for registration.

Grid convention: a slab grid with voxel size `um` has voxel i at level-0 position p = (i + 0.5) * um / l0_um - 0.5
(per axis) and shape floor(level0_shape * l0_um / um). Pyramid levels follow the same convention (OME-Zarr translation
of level k = (f - 1) / 2 level-0 voxels for downsampling factor f).
Axes: zarr (z, y, x) == atlas (AP, DV, ML); x is the slab thickness, the only axis that may be mirrored (flip)."""
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi
from skimage import filters, morphology

PREP_UM = 16.0                       # preprocessing resolution: pyramid level closest to this
OPEN_UM = {"540": 150.0, "630": 115.0}   # bright-object removal: disk diameter (plaques in 540, microglia in 630)


def _multiscales(g):
    attrs = g.attrs.asdict()
    ms = attrs["ome"]["multiscales"] if "ome" in attrs else attrs["multiscales"]
    return ms[0]


class Slab:
    """Fused OME-Zarr of one slab. ch_540: channel with tissue autofluorescence (+ plaques), ch_630: even channel
    (microglia) used for the tissue mask."""

    def __init__(self, zarr_path, ch_540=0, ch_630=1, prep_um=PREP_UM):
        self.path = Path(zarr_path)
        self.ch_540, self.ch_630 = int(ch_540), int(ch_630)
        g = zarr.open_group(str(self.path), mode="r")
        ms = _multiscales(g)
        self.level_paths = [d["path"] for d in ms["datasets"]]
        self.level_um = []
        for d in ms["datasets"]:
            sc = [t["scale"] for t in d["coordinateTransformations"] if t["type"] == "scale"][0][-3:]
            if not np.allclose(sc, sc[0], rtol=1e-3):
                raise ValueError(f"{self.path}: level {d['path']} is not isotropic ({sc})")
            self.level_um.append(float(sc[0]))
        self.l0_um = self.level_um[0]
        self.l0_shape = tuple(int(n) for n in g[self.level_paths[0]].shape[-3:])
        self.prep_level = int(np.argmin([abs(np.log(u / prep_um)) for u in self.level_um]))
        self.prep_um = self.level_um[self.prep_level]

    def grid_shape(self, um):
        return tuple(int(np.floor(n * self.l0_um / um)) for n in self.l0_shape)

    def load_prep_level(self):
        """Preprocessing pyramid level as float32 (c, z, y, x)."""
        a = zarr.open_group(str(self.path), mode="r")[self.level_paths[self.prep_level]]
        v = np.asarray(a[0] if a.ndim == 5 else a[:])
        return (v[None] if v.ndim == 3 else v).astype(np.float32)

    def resample_prep(self, v, um, order=1):
        """Exact, anti-aliased resampling of a prep-level volume onto the `um` slab grid."""
        f = um / self.prep_um
        if f > 1:
            v = ndi.gaussian_filter(v, 0.45 * f)
        idx = [(np.arange(n) + 0.5) * f - 0.5 for n in self.grid_shape(um)]
        zz, yy, xx = np.meshgrid(*idx, indexing="ij")
        return ndi.map_coordinates(v, [zz, yy, xx], order=order, mode="nearest").astype(np.float32)

    def resample_grid(self, v, um_from, um_to, order=1):
        """Resample an array on one slab grid onto another (e.g. 50 um mask -> 20 um)."""
        f = um_to / um_from
        idx = [(np.arange(n) + 0.5) * f - 0.5 for n in self.grid_shape(um_to)]
        zz, yy, xx = np.meshgrid(*idx, indexing="ij")
        return ndi.map_coordinates(v.astype(np.float32), [zz, yy, xx], order=order, mode="nearest")

    def l0_to_grid(self, p, um):
        """level-0 voxel (z, y, x) -> continuous index on the `um` slab grid."""
        return (np.asarray(p, float) + 0.5) * self.l0_um / um - 0.5

    def grid_to_l0(self, i, um):
        return (np.asarray(i, float) + 0.5) * um / self.l0_um - 0.5

    def registration_images(self, lvl, mask, um):
        """540 background (plaques removed) and 630 background (microglia removed) on the `um` grid."""
        out = []
        for ch, key in ((self.ch_540, "540"), (self.ch_630, "630")):
            px = int(round(OPEN_UM[key] / self.prep_um))
            out.append(robust_norm(self.resample_prep(remove_bright_objects(lvl[ch], px), um), mask, 1, 99))
        return out


def remove_bright_objects(v, size_px):
    """Grey opening per sagittal plane (prep-level px) with a DISK footprint: removes bright objects smaller than
    size_px (plaques, microglia), keeps the tissue background. A square footprint leaves blocky edges that act as
    false structure at 20 um, hence disk + median + light Gaussian."""
    fp = morphology.disk(size_px // 2)[:, :, None].astype(bool)
    op = ndi.grey_opening(v, footprint=fp)
    return ndi.gaussian_filter(ndi.median_filter(op, size=(3, 3, 1)), (1.0, 1.0, 0))


def tissue_mask(v, valid):
    """Tissue vs agarose/background from the even channel (630), on the 50 um grid."""
    s = ndi.gaussian_filter(v, 1.0)
    lg = np.log1p(s)
    thr = filters.threshold_otsu(lg[valid])
    m = (lg > thr) & valid
    m = morphology.remove_small_objects(m, max_size=500)
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n > 1:  # keep components >2% of tissue (drops specks, keeps detached cerebellum etc.)
        sz = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sz > 0.02 * sz.sum()))
    return m


def robust_norm(v, mask, lo=1, hi=99):
    a, b = np.percentile(v[mask], [lo, hi])
    return np.clip((v - a) / (b - a + 1e-6), 0, 1) * mask
