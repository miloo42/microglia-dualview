"""Register highres dualview FOVs to the lowres overview, as a library to call
from a notebook.

Each highres FOV is downsampled to the lowres voxel size and located in the
lowres volume by phase cross correlation. The result (translation of every FOV
in lowres micrometer coordinates) is written to a txt file, which
view_registered_fovs.py shows in napari.

Usage from a notebook:

    import sys
    sys.path.append(r"<repo>\\fov-registration")
    import fov_registration as fr

    results = fr.register_fovs(lowres_zarr, {"fov0": fov_zarr}, lowres_voxel_um=4.1012,
                               highres_voxel_um=0.512, out_txt=txt_path)

Translations are in lowres micrometers (lowres level-0 voxel 0 centered at 0),
so a highres coordinate maps to the lowres as
    lowres_um = highres_voxel * voxel_um + translate_um
"""
from pathlib import Path

import dask.array as da
import numpy as np

COLUMNS = ['name', 'path', 'voxel_um', 'tz', 'ty', 'tx', 'sz', 'sy', 'sx', 'downsample', 'pearson_r']


def center_pad_nd(fov, target_shape):
    pad_widths = []
    for fov_dim, target_dim in zip(fov.shape, target_shape):
        total_pad = target_dim - fov_dim
        pad_before = total_pad // 2
        pad_widths.append((pad_before, total_pad - pad_before))

    return np.pad(fov, pad_widths, mode='constant')


def load_lowres(lowres_zarr, level=1, channel=0):
    """Load one level of the lowres OME-Zarr pyramid ((t, c, z, y, x) levels 0, 1, 2, ...)."""
    lowres = da.from_zarr(str(Path(lowres_zarr) / str(level)))[0, channel, ...]
    return lowres.compute().astype(np.float32)


def register_fov(lowres, fov_path, lowres_voxel_um, highres_voxel_um, level=1):
    """Locate one isotropic highres FOV (z, y, x zarr) in the lowres array of the given level.

    lowres_voxel_um is the (isotropic) voxel size of lowres level 0; the voxel
    size is not read from the zarr metadata, which is not always correct.
    """
    from skimage.registration import phase_cross_correlation

    # lowres voxel size at the registration level; voxel i of level k is centered at
    # (i * 2**k + (2**k - 1) / 2) * lowres_voxel_um in lowres level-0 coordinates
    level_factor = 2 ** level
    lowres_reg_voxel = lowres_voxel_um * level_factor
    lowres_reg_offset = (level_factor - 1) / 2 * lowres_voxel_um

    fov = da.from_zarr(str(fov_path))
    factor = int(round(lowres_reg_voxel / highres_voxel_um))
    fov_ds = da.coarsen(np.mean, fov, {0: factor, 1: factor, 2: factor}, trim_excess=True)
    fov_ds = fov_ds.compute().astype(np.float32)
    padded_fov = center_pad_nd(fov_ds, lowres.shape)
    shift, _, _ = phase_cross_correlation(lowres, padded_fov, upsample_factor=4)

    # first downsampled voxel position in the lowres registration level (voxel units)
    start = np.array([(F - f) // 2 for F, f in zip(lowres.shape, fov_ds.shape)]) + shift
    # translation of highres voxel 0 (center) in lowres micrometers:
    # center of downsampled voxel 0 is (factor - 1) / 2 highres voxels away from highres voxel 0
    translate_um = start * lowres_reg_voxel + lowres_reg_offset - (factor - 1) / 2 * highres_voxel_um

    # quality check: correlation between the placed fov and the lowres it lands on
    lo = np.round(start).astype(int)
    hi = lo + np.array(fov_ds.shape)
    if np.all(lo >= 0) and np.all(hi <= np.array(lowres.shape)):
        crop = lowres[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        r = np.corrcoef(crop.ravel(), fov_ds.ravel())[0, 1]
    else:
        r = np.nan  # fov placed partly outside the lowres volume

    return dict(path=str(fov_path), voxel_um=highres_voxel_um,
                tz=translate_um[0], ty=translate_um[1], tx=translate_um[2],
                sz=shift[0], sy=shift[1], sx=shift[2], downsample=factor, pearson_r=r)


def register_fovs(lowres_zarr, fovs, lowres_voxel_um, highres_voxel_um,
                  channel=0, level=1, out_txt=None):
    """Register every FOV in fovs ({name: zarr path}) and optionally write out_txt.

    level 1 (2x downsampled lowres, ~8.2 um) is the default registration level.
    Returns a list of result dicts (one per FOV, keys as in COLUMNS).
    """
    lowres = load_lowres(lowres_zarr, level, channel)
    print('lowres', lowres.shape, f'{lowres_voxel_um * 2 ** level:.4f} um')

    results = []
    for name, fov_path in fovs.items():
        print(name, fov_path)
        res = dict(name=name, **register_fov(lowres, fov_path, lowres_voxel_um,
                                             highres_voxel_um, level))
        print('  shift', np.round([res['sz'], res['sy'], res['sx']], 2),
              'translate_um', np.round([res['tz'], res['ty'], res['tx']], 2),
              'pearson r', round(res['pearson_r'], 3))
        results.append(res)

    if out_txt is not None:
        write_registration(out_txt, results, lowres_zarr, lowres_voxel_um, channel, level)
    return results


def write_registration(txt_path, results, lowres_zarr, lowres_voxel_um, channel, level):
    with open(txt_path, 'w') as f:
        f.write(f'# lowres_zarr = {lowres_zarr}\n')
        f.write(f'# lowres_voxel_um = {lowres_voxel_um}\n')
        f.write(f'# lowres_channel = {channel}\n')
        f.write(f'# registration_level = {level}\n')
        f.write('\t'.join(COLUMNS) + '\n')
        for res in results:
            f.write('\t'.join(str(res[c]) for c in COLUMNS) + '\n')
    print('written', txt_path)


def read_registration(txt_path):
    """Return (header dict, list of row dicts with string values) of a registration txt."""
    header = {}
    rows = []
    columns = None
    with open(txt_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('#'):
                key, value = line[1:].split('=', 1)
                header[key.strip()] = value.strip()
            elif columns is None:
                columns = line.split('\t')
            else:
                rows.append(dict(zip(columns, line.split('\t'))))
    return header, rows
