"""Turn boxes drawn in napari on a fused OME-ZARR into Micro-Manager pos.pos files (one per ROI).

The fused volume is mapped linearly onto the stage coordinates of the original acquisition:
  - voxel (0, 0, 0) is the minimum corner of the volume covered by the original acquisition
  - one voxel is `res` um along every axis (the isotropic target_res of the processing)
  - no crop, no resizing; stitching shifts are ignored (~100 um at the far end)
  - along the scan axis, voxel 0 is the scan start for path A (RightCam) and the scan end for path B (LeftCam),
    as written by the pipeline's deskew

pos.pos convention (ASIdiSPIM plugin, MM 1.4)
---------------------------------------------
Y (XYStage) and Z (ZStage) of a position are the centre of the camera field of view, as made by the
plugin's XYZ grid (the grid limits are FOV centres of the first and last tile). X is the centre of the
stage scan (scan = X -/+ numSlices * step * sqrt(2) / 2), unless 'scan from start position' is set in the
plugin settings, in which case X is the scan start. The mode is read from the scan start that the controller
logged in metadata.txt, and the new pos.pos files are written in the same mode.

Tiles per ROI
-------------
Every ROI is covered by a grid of Y/Z tiles with --overlap (default 10 %), centred on the ROI
(FOV Y = camera width, FOV Z = camera height / sqrt(2)). With --single-tile every ROI becomes one position
centred on the ROI instead; parts of the ROI outside that field of view are not acquired, which is reported.

Scan start and number of slices
-------------------------------
Each page is a 45 deg plane. A voxel at depth u (um below the top of the tile's field of view) and at
stage-equivalent position X_eq is imaged
  - by view A when the stage is at X = X_eq - u        (checked against util.get_affine_transform)
  - by view B when the stage is at X = X_eq - (F - u)  (sheet tilted the other way, F = FOV in Z;
                                                         assumes both sheets cross at the FOV centre)
For each row of Z tiles, the scan covers the box for the views given with --views (default both), so that
the two views overlap over the whole ROI. The number of slices is set by the longest scan any row needs.

Which zarr axis belongs to which stage axis is decided by matching the physical size of the zarr
(shape * res) to the extents of the original acquisition, and printed for checking.
Expected for this pipeline: zarr (z, y, x) = (XYStage X = scan, XYStage Y, ZStage).

Drawing the boxes (in any napari session)
-----------------------------------------
1. Open the zarr in napari (drag & drop the .zarr folder, or `viewer.open(path)`): the built-in reader
   shows it in voxel units. If you open a single pyramid level instead, pass --level.
2. Add a Shapes layer and draw rectangles. A rectangle is flat in the axis it was drawn across:
   - one rectangle alone -> ROI spans the full volume along its flat axis
   - rectangles that overlap in the axes they share are merged into one ROI (their joint bounding box),
     e.g. draw the same outline on the first and on the last slice of the ROI, or one rectangle in the
     XY view and one in an orthogonal view.
3. Save the Shapes layer as .csv (File > Save Selected Layer(s)...).

Usage
-----
In napari: the 'Slab atlas' widget (ROIs section) calls write_pos_files() on its Shapes layer.
Command line:
python -m napari_slab_atlas.pospos <fused.zarr> <raw_folder> <shapes.csv> [--out DIR] [--overlap 0.1]
                              [--single-tile] [--views AB] [--margin 0] [--res 4.1012] [--pix-um 1.45]
                              [--level 0] [--units voxel] [--flip Y Z]

Every ROI is a separate acquisition. For each ROI this writes <out>/ROI<i>.pos (grid of Y/Z tiles covering
the ROI, or one position with --single-tile) and a summary file with the number of slices to set in the acquisition panel, since the scan length
(numSlices) is a global acquisition setting and not part of pos.pos. The step size follows from --mfs
(MFS = stage X delta per slice = step size * sqrt(2)).
<out>/ROIs.tsv lists name, .pos file, number of slices and step size per ROI for tools/acquire_rois.bsh.
"""

import argparse
import copy
import csv
import glob
import itertools
import json
import os
import re

import numpy as np
import tifffile

STAGE_AXES = ['X', 'Y', 'Z']  # XYStage X (scan), XYStage Y, ZStage
ISO_RE = re.compile(r"_iso(\d+p\d+|\d+)")


# ---------------------------------------------------------------------------------------------------------
# inputs

def zarr_level_shapes(zarr_fn):
    """Spatial shapes (last 3 axes) of every pyramid level, and the .zattrs scale of level 0"""
    zattrs_fn = os.path.join(zarr_fn, '.zattrs')
    if os.path.isfile(zattrs_fn):
        with open(zattrs_fn) as f:
            ms = json.load(f)['multiscales'][0]
    else:
        with open(os.path.join(zarr_fn, 'zarr.json')) as f:
            ms = json.load(f)['attributes']['ome']['multiscales'][0]
    shapes = []
    for ds in ms['datasets']:
        level_fn = os.path.join(zarr_fn, ds['path'])
        if os.path.isfile(os.path.join(level_fn, '.zarray')):
            with open(os.path.join(level_fn, '.zarray')) as f:
                shape = json.load(f)['shape']
        else:
            with open(os.path.join(level_fn, 'zarr.json')) as f:
                shape = json.load(f)['shape']
        shapes.append(np.array(shape[-3:]))
    scale0 = [t['scale'] for t in ms['datasets'][0]['coordinateTransformations'] if t['type'] == 'scale'][0]
    return shapes, np.array(scale0[-3:], dtype=float)


def find_res(zarr_fn, zattrs_scale, res=None):
    """Isotropic voxel size in um: --res, else an isotropic .zattrs scale, else _iso<res> of the .xml"""
    if res is not None:
        return res, '--res'
    if np.allclose(zattrs_scale, zattrs_scale[0], rtol=1e-3) and not np.allclose(zattrs_scale, 1):
        return float(zattrs_scale[0]), '.zattrs scale'
    # <dataset>_out/fused/<name>.zarr -> <dataset>_out/*_iso4p1012_*.xml
    out_folder = os.path.dirname(os.path.dirname(os.path.abspath(zarr_fn)))
    for xml_fn in glob.glob(os.path.join(out_folder, '*.xml')):
        m = ISO_RE.search(os.path.basename(xml_fn))
        if m is not None:
            return float(m.group(1).replace('p', '.')), os.path.basename(xml_fn)
    raise ValueError(f"Cannot determine the voxel size of {zarr_fn} (.zattrs scale {zattrs_scale} is not "
                     f"isotropic and no *_iso*.xml in {out_folder}), pass --res")


def read_raw(raw_folder, cam_width=None, cam_height=None, pix_um=None, camera=None, scan_from_start=None):
    """pos.pos (raw json), stage positions (n, 3) in X Y Z order, and acquisition settings

    The settings come from the Micro-Manager metadata of the first .tif. Without .tif files the folder needs
    AcqSettings.txt (the plugin's SPIMAcqSettings json) and the camera size, pixel size and camera must be given.
    Given values override the metadata."""
    pos_fns = glob.glob(os.path.join(raw_folder, '*.pos'))
    if len(pos_fns) != 1:
        raise ValueError(f"Expected one .pos file in {raw_folder}, found {pos_fns}")
    with open(pos_fns[0]) as f:
        pos_pos = json.load(f)

    stage = []
    for p in pos_pos['POSITIONS']:
        devs = {d['DEVICE'].split(':')[0]: d for d in p['DEVICES']}
        # ZStage stores its single axis in the 'X' field
        stage.append([devs['XYStage']['X'], devs['XYStage']['Y'], devs['ZStage']['X']])
    stage = np.array(stage, dtype=float)

    tif_fns = sorted(glob.glob(os.path.join(raw_folder, '*.tif')))
    acq_fn = os.path.join(raw_folder, 'AcqSettings.txt')
    if len(tif_fns) > 0:
        with tifffile.TiffFile(tif_fns[0]) as tif:
            summary = tif.micromanager_metadata['Summary']
            try:
                cam_meta = tif.micromanager_metadata['DisplaySettings'][0]['Name']
            except (KeyError, IndexError, TypeError):
                cam_meta = 'RightCam'  # path A, as in metadata.assemble_mdat
        spim = summary['SPIMAcqSettings']
        if isinstance(spim, str):
            spim = json.loads(spim)
        cam_width = cam_width or int(summary['Width'])
        cam_height = cam_height or int(summary['Height'])
        pix_um = pix_um or float(summary['PixelSize_um'])
        camera = camera or cam_meta
        mtxt_fns = glob.glob(tif_fns[0].split('.ome.tif')[0].split('.tif')[0] + '_metadata.txt')
    elif os.path.isfile(acq_fn):
        with open(acq_fn) as f:
            spim = json.load(f)
        missing = [n for n, v in (('camera width', cam_width), ('camera height', cam_height), ('pixel size', pix_um))
                   if not v]
        if missing:
            raise ValueError(f"No .tif in {raw_folder}: give the {', '.join(missing)} (only AcqSettings.txt found)")
        camera = camera or ('RightCam' if spim.get('firstSideIsA', True) else 'LeftCam')
        mtxt_fns = glob.glob(os.path.join(raw_folder, '*metadata.txt'))
    else:
        raise ValueError(f"No .tif files and no AcqSettings.txt in {raw_folder}")

    # scan start logged by the controller for the first position -> centred scan or scan from start
    scan_len = int(spim['numSlices']) * np.sqrt(2) * float(spim['stepSizeUm'])
    scan_frac = -0.5  # plugin default: X is the scan centre
    if scan_from_start is not None:
        scan_frac = 0 if scan_from_start else -0.5
    elif len(mtxt_fns) > 0:
        with open(mtxt_fns[0], encoding='utf-8', errors='ignore') as f:
            head = f.read(2_000_000)
        start = re.search(r'ScanFastAxisStartPosition\(mm\)": "([-\d.]+)"', head)
        stop = re.search(r'ScanFastAxisStopPosition\(mm\)": "([-\d.]+)"', head)
        if start is not None and stop is not None:
            start, stop = float(start.group(1)) * 1000, float(stop.group(1)) * 1000
            if stop < start:
                raise NotImplementedError("Scan in negative direction is not supported")
            frac = (start - stage[0, 0]) / scan_len
            scan_frac = min([-0.5, 0], key=lambda c: abs(c - frac))
            if abs(frac - scan_frac) > 0.05:
                print(f"WARNING: logged scan start {start:.1f} um does not match pos.pos X {stage[0, 0]:.1f} um "
                      f"as centre or start of a {scan_len:.1f} um scan")
    else:
        print("WARNING: no metadata.txt found, assuming the pos.pos X is the scan centre")

    acq = {
        'pos_fn': pos_fns[0],
        'scan_frac': scan_frac,
        'camera': camera,
        'width': int(cam_width),
        'height': int(cam_height),
        'pix_um': float(pix_um),
        'num_slices': int(spim['numSlices']),
        'step_um': float(spim['stepSizeUm']),
    }
    return pos_pos, stage, acq


def read_napari_shapes_csv(csv_fn):
    """napari Shapes .csv -> list of (n_vertices, 3) arrays (last 3 axes)"""
    shapes = {}
    with open(csv_fn, newline='') as f:
        reader = csv.DictReader(f)
        axis_cols = sorted([c for c in reader.fieldnames if c.startswith('axis-')], key=lambda c: int(c[5:]))
        if len(axis_cols) < 3:
            raise ValueError(f"{csv_fn} has {len(axis_cols)} axis columns; draw the shapes on the 3D volume")
        for row in reader:
            shapes.setdefault(int(row['index']), []).append([float(row[c]) for c in axis_cols[-3:]])
    return [np.array(shapes[k]) for k in sorted(shapes)]


# ---------------------------------------------------------------------------------------------------------
# geometry

def match_axes(zarr_extent, stage_extent):
    """perm[i] = stage axis of zarr axis i, chosen by matching physical extents"""
    best = None
    for perm in itertools.permutations(range(3)):
        err = np.sum(np.abs(np.log(zarr_extent / stage_extent[list(perm)])))
        if best is None or err < best[0]:
            best = (err, perm)
    return list(best[1])


def shapes_to_boxes(shapes, vol_shape, tol=0.5):
    """Merge planar shapes into 3D boxes (in voxels of the zarr axes)

    Two shapes belong to the same box if, along every axis, their extents are compatible:
    both span the axis -> intervals overlap; one is flat -> its coordinate lies inside the other's interval;
    both flat -> no constraint (e.g. top and bottom face).
    Axes along which a box is still flat span the full volume.
    """
    lo = np.array([s.min(axis=0) for s in shapes])
    hi = np.array([s.max(axis=0) for s in shapes])
    flat = (hi - lo) < tol

    parent = list(range(len(shapes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(shapes)), 2):
        ok = True
        for a in range(3):
            if flat[i, a] and flat[j, a]:
                continue
            if hi[i, a] + tol < lo[j, a] or hi[j, a] + tol < lo[i, a]:
                ok = False
                break
        if ok:
            parent[root(j)] = root(i)

    boxes = []
    for r in sorted(set(root(i) for i in range(len(shapes)))):
        members = [i for i in range(len(shapes)) if root(i) == r]
        b_lo = lo[members].min(axis=0)
        b_hi = hi[members].max(axis=0)
        full = (b_hi - b_lo) < tol
        b_lo[full] = 0
        b_hi[full] = vol_shape[full]
        b_lo = np.clip(b_lo, 0, vol_shape)
        b_hi = np.clip(b_hi, 0, vol_shape)
        boxes.append({'shapes': members, 'lo': b_lo, 'hi': b_hi, 'full_axes': np.where(full)[0]})
    return boxes


def tile_axis(lo, hi, fov, overlap):
    """Lower FOV edges of tiles covering [lo, hi], centred on the interval"""
    length = hi - lo
    step = fov * (1 - overlap)
    n = 1 if length <= fov else int(np.ceil((length - fov) / step)) + 1
    total = fov + (n - 1) * step
    start = (lo + hi) / 2 - total / 2
    return start + step * np.arange(n)


def make_pos_pos(template, xs, ys, zs):
    """pos.pos dict with a grid of Y (inner) and Z (outer) tiles, labelled Pos_<z>_<y> like the original
    xs holds the scan position (as in pos.pos) of each Z row"""
    out = {k: v for k, v in template.items() if k != 'POSITIONS'}
    out['POSITIONS'] = []
    p0 = template['POSITIONS'][0]
    for zi, z in enumerate(zs):
        for yi, y in enumerate(ys):
            p = copy.deepcopy(p0)
            for d in p['DEVICES']:
                if d['DEVICE'].startswith('XYStage'):
                    d['X'] = round(float(xs[zi]), 4)
                    d['Y'] = round(float(y), 4)
                elif d['DEVICE'].startswith('ZStage'):
                    d['X'] = round(float(z), 4)
            p['LABEL'] = f'Pos_{zi}_{yi}'
            p['GRID_ROW'] = 0
            p['GRID_COL'] = 0
            out['POSITIONS'].append(p)
    return out


# ---------------------------------------------------------------------------------------------------------


MFS_FILE = r'C:\Users\Lightsheet\Desktop\MFS_X_delta.txt'


def write_pos_files(zarr_fn, raw_folder, shapes, out_dir, *, overlap=0.1, single_tile=False, views='AB', mfs=0.5,
                    mfs_file=MFS_FILE, margin=0.0, res=None, pix_um=None, cam_width=None, cam_height=None,
                    camera=None, scan_from_start=None, level=0, units='voxel', flip=(), source='', log=print):
    """Write <out_dir>/ROI<i>.pos, ROIs_summary.txt and ROIs.tsv for planar shapes drawn on the fused zarr.

    shapes: list of (n_vertices, 3) arrays (z, y, x) in voxels of pyramid `level` (units='voxel') or in the
    zarr's .zattrs units (units='zattrs'). Returns one dict per ROI (name, pos_file, num_slices, step_um,
    lo/hi in level-0 voxels, number of positions)."""
    # --- zarr
    level_shapes, zattrs_scale = zarr_level_shapes(zarr_fn)
    vol_shape = level_shapes[0]
    res, res_src = find_res(zarr_fn, zattrs_scale, res)
    zarr_extent = vol_shape * res

    # --- raw acquisition
    pos_pos, stage, acq = read_raw(raw_folder, cam_width, cam_height, pix_um, camera, scan_from_start)
    pix = acq['pix_um']
    scan_step = np.sqrt(2) * acq['step_um']  # stage step between pages
    fov = np.array([np.nan,
                    acq['width'] * pix,                 # Y: camera width
                    acq['height'] * pix / np.sqrt(2)])  # Z: in-sheet camera axis projected on depth
    # min corner of the acquired volume: pos.pos Y and Z are FOV centres, X is scan centre or start
    stage_lo = stage.min(axis=0) - np.array([-acq['scan_frac'] * acq['num_slices'] * scan_step,
                                             fov[1] / 2, fov[2] / 2])
    stage_extent = np.array([acq['num_slices'] * scan_step + fov[2],  # + 45 deg sheet footprint
                             np.ptp(stage[:, 1]) + fov[1],
                             np.ptp(stage[:, 2]) + fov[2]])

    perm = match_axes(zarr_extent, stage_extent)
    flip = list(flip) + (['X'] if acq['camera'] == 'LeftCam' else [])
    sign = np.array([-1 if STAGE_AXES[perm[i]] in flip else 1 for i in range(3)])

    log(f"zarr     : {zarr_fn}")
    log(f"pos.pos  : {acq['pos_fn']} ({len(stage)} positions)")
    log(f"voxel    : {res} um (from {res_src})")
    log(f"camera   : {acq['width']}x{acq['height']} px, {pix} um/px -> FOV Y {fov[1]:.1f} um, Z {fov[2]:.1f} um")
    scan_mode = 'pos.pos X = scan centre' if acq['scan_frac'] < 0 else 'pos.pos X = scan start'
    log(f"scan     : {acq['num_slices']} slices x {scan_step:.4f} um, {scan_mode}, camera {acq['camera']}")

    # MFS of the ROI acquisitions: stage X delta per slice, from the list of allowed values
    with open(mfs_file) as f:
        mfs_values = np.array([float(line) for line in f if line.strip()])
    mfs_asked = mfs
    mfs = float(mfs_values[np.argmin(np.abs(mfs_values - mfs))])
    roi_step_um = mfs / np.sqrt(2)
    log(f"ROI MFS  : {mfs} um per slice (asked {mfs_asked}, allowed {mfs_values.tolist()}), "
        f"step size {roi_step_um:.7f} um")
    log("axis matching (zarr size vs acquired size):")
    for i, name in enumerate('zyx'):
        s = perm[i]
        log(f"  zarr {name}: {vol_shape[i]:5d} vox = {zarr_extent[i]:8.1f} um  <->  stage {STAGE_AXES[s]}: "
            f"{stage_extent[s]:8.1f} um  ({zarr_extent[i] / stage_extent[s] - 1:+.1%})"
            f"{'  flipped' if sign[i] < 0 else ''}")
    if perm != [0, 1, 2]:
        log("  WARNING: axis order differs from the expected zarr (z, y, x) = stage (X, Y, Z)")

    # voxel index (level 0) along zarr axis i -> stage coordinate of stage axis perm[i]
    def vox_to_stage(v, i):
        s = perm[i]
        if sign[i] > 0:
            return stage_lo[s] + v * res
        return stage_lo[s] + stage_extent[s] - v * res

    # --- shapes -> boxes (in level-0 voxels)
    shapes = [np.asarray(s, float)[:, -3:] for s in shapes]
    if units == 'zattrs':
        shapes = [s / zattrs_scale for s in shapes]
    else:
        factor = vol_shape / level_shapes[level]
        shapes = [s * factor for s in shapes]
    for k, s in enumerate(shapes):
        if np.any(s < -0.5) or np.any(s > vol_shape + 0.5):
            log(f"WARNING: shape {k} lies (partly) outside the volume {vol_shape} - check --level / --units")
    boxes = shapes_to_boxes(shapes, vol_shape)
    log(f"{len(shapes)} shapes -> {len(boxes)} ROIs")

    os.makedirs(out_dir, exist_ok=True)

    summary = [f"zarr: {zarr_fn}", f"raw pos.pos: {acq['pos_fn']}", f"shapes: {source}",
               f"voxel {res} um, axis order zarr(z,y,x) = stage({','.join(STAGE_AXES[p] for p in perm)}), "
               f"origin X Y Z = {stage_lo.tolist()} um, flipped: {flip}, camera {acq['camera']}, {scan_mode}",
               ("one position per ROI" if single_tile else f"tile overlap {overlap}")
               + f", views {views}, margin {margin} um, "
               f"ROI acquisitions: MFS {mfs} um per slice, stepSizeUm {roi_step_um:.7f} "
               f"(original: stepSizeUm {acq['step_um']}, scan step {scan_step:.4f} um)", ""]

    tsv = ['name\tpos_file\tnum_slices\tstep_um']
    rois = []
    for r, box in enumerate(boxes):
        # stage bounds of the box, ordered X Y Z
        lo = np.zeros(3)
        hi = np.zeros(3)
        for i in range(3):
            a, b = vox_to_stage(box['lo'][i], i), vox_to_stage(box['hi'][i], i)
            lo[perm[i]], hi[perm[i]] = min(a, b) - margin, max(a, b) + margin

        # tiles (lower FOV edges): a grid covering the ROI, or one tile centred on it
        cropped = []
        if single_tile:
            ys = np.array([(lo[1] + hi[1]) / 2 - fov[1] / 2])
            zs = np.array([(lo[2] + hi[2]) / 2 - fov[2] / 2])
            cropped = [f"{ax} {hi[s] - lo[s]:.0f} > FOV {fov[s]:.0f} um" for s, ax in [(1, 'Y'), (2, 'Z')]
                       if hi[s] - lo[s] > fov[s]]
        else:
            ys = tile_axis(lo[1], hi[1], fov[1], overlap)
            zs = tile_axis(lo[2], hi[2], fov[2], overlap)

        # scan of each Z row: depth u of the box inside the row's FOV shifts the stage X by
        # -u (view A) or -(F - u) (view B)
        u_min = np.clip(lo[2] - zs, 0, fov[2])
        u_max = np.clip(hi[2] - zs, 0, fov[2])
        need = {'A': (lo[0] - u_max, hi[0] - u_min),
                'B': (lo[0] - (fov[2] - u_min), hi[0] - (fov[2] - u_max))}
        xs = np.min([need[v][0] for v in views], axis=0)
        scan_len = np.max([need[v][1] for v in views], axis=0) - xs
        n_slices = int(np.ceil(scan_len.max() / mfs))

        # pos.pos values: FOV centres in Y and Z, scan centre (or start) in X
        pos_x = xs - acq['scan_frac'] * n_slices * mfs
        pos_y = ys + fov[1] / 2
        pos_z = zs + fov[2] / 2

        fn = os.path.join(out_dir, f'ROI{r}.pos')
        with open(fn, 'w') as f:
            json.dump(make_pos_pos(pos_pos, pos_x, pos_y, pos_z), f, indent=3)

        full = ', '.join('zyx'[a] for a in box['full_axes'])
        lines = [f"ROI{r}: {fn}",
                 f"  from shapes {box['shapes']}" + (f" (full extent along zarr {full})" if full else ""),
                 f"  voxels (z,y,x) lo {np.round(box['lo']).astype(int).tolist()} hi {np.round(box['hi']).astype(int).tolist()}",
                 f"  stage X {lo[0]:.1f} .. {hi[0]:.1f}  Y {lo[1]:.1f} .. {hi[1]:.1f}  Z {lo[2]:.1f} .. {hi[2]:.1f} um",
                 f"  {len(zs)} (Z) x {len(ys)} (Y) = {len(zs) * len(ys)} positions"
                 + (f"   WARNING: ROI cropped to one FOV ({', '.join(cropped)})" if cropped else ""),
                 f"  pos.pos Y {', '.join(f'{y:.1f}' for y in pos_y)}",
                 *[f"  Z row {zi}: pos.pos Z {pos_z[zi]:.1f}, X {pos_x[zi]:.1f}; scan X {xs[zi]:.1f} .. "
                   f"{xs[zi] + scan_len[zi]:.1f} ("
                   + ', '.join(f"view {v} {need[v][0][zi]:.1f} .. {need[v][1][zi]:.1f}" for v in views) + ")"
                   for zi in range(len(zs))],
                 f"  -> set Number of slices = {n_slices}, step size {roi_step_um:.7f} um "
                 f"(MFS {mfs}, scan {n_slices * mfs:.1f} um)"]
        log('\n'.join(lines))
        summary += lines + ['']
        tsv.append(f"ROI{r}\t{os.path.abspath(fn)}\t{n_slices}\t{roi_step_um:.7f}")
        rois.append(dict(name=f'ROI{r}', pos_file=os.path.abspath(fn), num_slices=n_slices, step_um=roi_step_um,
                         lo=box['lo'], hi=box['hi'], positions=len(zs) * len(ys)))

    with open(os.path.join(out_dir, 'ROIs_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    with open(os.path.join(out_dir, 'ROIs.tsv'), 'w') as f:
        f.write('\n'.join(tsv) + '\n')
    log(f"written to {out_dir}")
    return rois


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('zarr', help='fused OME-ZARR')
    ap.add_argument('raw_folder', help='raw data folder with pos.pos and the .tif files')
    ap.add_argument('shapes_csv', help='napari Shapes layer saved as .csv')
    ap.add_argument('--out', default=None, help='output folder (default: <shapes_csv>_pos next to the csv)')
    ap.add_argument('--overlap', type=float, default=0.1, help='tile overlap fraction in Y and Z (default 0.1)')
    ap.add_argument('--single-tile', action='store_true',
                    help='one position per ROI, centred on it (ROI cropped to one field of view)')
    ap.add_argument('--views', choices=['A', 'B', 'AB'], default='AB',
                    help='views whose slanted sheets must cover the whole ROI (default: both)')
    ap.add_argument('--mfs', type=float, default=0.5,
                    help='MFS of the ROI acquisition = stage X delta per slice in um, snapped to the nearest value '
                         'in --mfs-file (default 0.5 -> 0.5127); step size = MFS / sqrt(2)')
    ap.add_argument('--mfs-file', default=MFS_FILE,
                    help='text file with the allowed MFS / X delta values, one per line')
    ap.add_argument('--margin', type=float, default=0, help='margin added around each ROI, in um')
    ap.add_argument('--res', type=float, default=None, help='voxel size of the zarr in um')
    ap.add_argument('--pix-um', type=float, default=None,
                    help='camera pixel size in the sample, in um (default: PixelSize_um of the .tif)')
    ap.add_argument('--cam-width', type=int, default=None, help='camera width in px (default: from the .tif)')
    ap.add_argument('--cam-height', type=int, default=None, help='camera height in px (default: from the .tif)')
    ap.add_argument('--camera', choices=['RightCam', 'LeftCam'], default=None,
                    help='camera of the overview (default: from the .tif, or path A = RightCam with AcqSettings.txt)')
    ap.add_argument('--scan-from-start', choices=['yes', 'no'], default=None,
                    help="pos.pos X is the scan start (default: from metadata.txt, else 'no' = scan centre)")
    ap.add_argument('--level', type=int, default=0, help='pyramid level the shapes were drawn on')
    ap.add_argument('--units', choices=['voxel', 'zattrs'], default='voxel',
                    help="shape coordinates in voxels (napari built-in reader) or in the zarr's .zattrs units")
    ap.add_argument('--flip', nargs='*', default=[], choices=STAGE_AXES[1:],
                    help='stage axes along which increasing voxel index means decreasing stage coordinate '
                         '(X is set from the camera: flipped for LeftCam)')
    args = ap.parse_args()
    out_dir = args.out or os.path.splitext(os.path.abspath(args.shapes_csv))[0] + '_pos'
    write_pos_files(args.zarr, args.raw_folder, read_napari_shapes_csv(args.shapes_csv), out_dir,
                    overlap=args.overlap, single_tile=args.single_tile, views=args.views, mfs=args.mfs,
                    mfs_file=args.mfs_file, margin=args.margin, res=args.res, pix_um=args.pix_um,
                    cam_width=args.cam_width, cam_height=args.cam_height, camera=args.camera,
                    scan_from_start=None if args.scan_from_start is None else args.scan_from_start == 'yes',
                    level=args.level,
                    units=args.units, flip=args.flip, source=args.shapes_csv)


if __name__ == '__main__':
    main()
