"""itk-elastix helpers. Fixed = slab, moving = Allen template (brainglobe convention), so the transform maps
slab coords -> atlas coords. Parameter maps start from elastix defaults (brainglobe's setup_parameter_object clears
them, which silently drops Registration/pyramid/Resampler keys). Physical units are mm, itk order (x, y, z)."""
import os
import re
import tempfile
from pathlib import Path

import itk
import numpy as np
from scipy import ndimage as ndi

ATLAS_SP = 0.05          # 50 um atlas spacing in mm


def to_itk(arr, origin=(0, 0, 0), dtype=np.float32, spacing=ATLAS_SP):
    im = itk.GetImageFromArray(np.ascontiguousarray(arr.astype(dtype)))
    im.SetSpacing((spacing,) * 3)
    im.SetOrigin(tuple(float(o) for o in origin))
    return im


def atlas_mask(template, thr_frac=0.05):
    m = template > thr_frac * np.percentile(template[template > 0], 99)
    return ndi.binary_fill_holes(ndi.binary_closing(m, iterations=2))


def prep_atlas(template, sigma=0.7):
    t = ndi.gaussian_filter(template, sigma)
    return t / np.percentile(t[t > 0], 99)


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def nmi(a, b, bins=32):
    """Deterministic normalised MI (Studholme), 1 = independent, 2 = identical."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    H = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    return (H(px) + H(py)) / H(p)


def param_map(po, kind, **over):
    pm = po.GetDefaultParameterMap(kind)
    base = dict(Metric=["AdvancedMattesMutualInformation"], NumberOfHistogramBins=["32"], ErodeMask=["false"],
                WriteResultImage=["true"], ResultImagePixelType=["float"], ImageSampler=["RandomCoordinate"],
                AutomaticTransformInitialization=["false"], DefaultPixelValue=["0"],
                RequiredRatioOfValidSamples=["0.05"], FinalBSplineInterpolationOrder=["1"], ResultImageFormat=["nii"])
    for k, v in {**base, **over}.items():
        pm[k] = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
    return pm


def multi_metric(n, weights, **over):
    """Overrides for a MultiMetric stage: MI(540) + MI(630) + MeanSquares(outline) [+ bending energy if 4 weights]."""
    mets = ["AdvancedMattesMutualInformation", "AdvancedMattesMutualInformation", "AdvancedMeanSquares",
            "TransformBendingEnergyPenalty"][:n]
    o = dict(Registration="MultiMetricMultiResolutionRegistration", Metric=mets,
             FixedImagePyramid=["FixedSmoothingImagePyramid"] * n, MovingImagePyramid=["MovingSmoothingImagePyramid"] * n,
             Interpolator=["LinearInterpolator"] * n, ImageSampler=["RandomCoordinate"] * n, UseRelativeWeights="true")
    for i, w in enumerate(weights):
        o[f"Metric{i}RelativeWeight"] = w
    o.update(over)
    return o


def run_elastix(po, images, init=None, log_dir=None):
    """images: (fixed 540, moving, fixed mask, moving mask), (fixed 630, moving, fixed mask),
    (fixed outline, moving outline, outline mask)[, (fixed 540, moving, fixed mask) for the bending-energy slot].
    The outline pair has no moving mask: the outline term must see atlas background.
    Returns (result image as numpy, transform ParameterObject)."""
    (FI, M, FM, MM), *extra = images
    R = itk.ElastixRegistrationMethod.New(FI, M)
    R.SetParameterObject(po)
    if init is not None:
        R.SetInitialTransformParameterObject(init)
    R.SetFixedMask(FM)
    R.SetMovingMask(MM)
    for fi, mi, fm in extra:
        R.AddFixedImage(fi)
        R.AddMovingImage(mi)
        R.AddFixedMask(fm)
    R.LogToConsoleOff()
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        R.SetOutputDirectory(str(log_dir))
        R.LogToFileOn()
    else:
        R.LogToFileOff()
    R.UpdateLargestPossibleRegion()
    return np.asarray(R.GetOutput()), R.GetTransformParameterObject()


def write_tp(tp, out):
    """TransformParameters.N.txt; the chain is the file order (no InitialTransformParameterFileName links)."""
    for f in Path(out).glob("TransformParameters.*.txt"):
        f.unlink()
    for i in range(tp.GetNumberOfParameterMaps()):
        m = tp.GetParameterMap(i)
        m["InitialTransformParameterFileName"] = ["NoInitialTransform"]
        itk.ParameterObject.WriteParameterFile(m, str(Path(out) / f"TransformParameters.{i}.txt"))


def clean_intermediates(out):
    for f in list(Path(out).glob("InitialTransformParameters.*.txt")) + list(Path(out).glob("result.*.nii")):
        f.unlink()


def load_tp(regdir):
    """Read TransformParameters.N.txt in order into one ParameterObject (maps composed in order)."""
    tp = itk.ParameterObject.New()
    files = sorted(Path(regdir).glob("TransformParameters.*.txt"), key=lambda f: int(f.stem.split(".")[1]))
    for f in files:
        # elastix writes "(InitialTransformParameterFileName)" without a value when the init came in as an object;
        # the parser rejects that line, so drop value-less lines (maps are composed by order in the object anyway)
        txt = re.sub(r"^\([A-Za-z]+\)\s*$", "", f.read_text(), flags=re.M)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write(txt)
        p = itk.ParameterObject.New()
        p.ReadParameterFile(tf.name)
        os.unlink(tf.name)
        tp.AddParameterMap(p.GetParameterMap(0))
    return tp


def subset_tp(tp, n):
    """First n maps of a ParameterObject (e.g. similarity only, similarity+affine)."""
    out = itk.ParameterObject.New()
    for i in range(n):
        out.AddParameterMap(tp.GetParameterMap(i))
    return out


def po_from_dicts(maps):
    po = itk.ParameterObject.New()
    for d in maps:
        m = po.GetDefaultParameterMap("rigid")
        m.clear()
        for k, v in d.items():
            m[k] = list(v)
        po.AddParameterMap(m)
    return po


def tp_to_dicts(tp):
    return [{k: list(v) for k, v in tp.GetParameterMap(i).items()} for i in range(tp.GetNumberOfParameterMaps())]


def moving_coords(tp, shape_zyx, spacing, origin=(0, 0, 0)):
    """Moving-space physical coords (mm, itk order x,y,z) of every voxel of a fixed grid, via transformix
    deformation field. Returns array (z,y,x,3)."""
    tp2 = itk.ParameterObject.New()
    for i in range(tp.GetNumberOfParameterMaps()):
        m = tp.GetParameterMap(i)
        m["Size"] = [str(s) for s in shape_zyx[::-1]]
        m["Spacing"] = [str(spacing)] * 3
        m["Origin"] = [str(o) for o in origin]
        m["Direction"] = ["1", "0", "0", "0", "1", "0", "0", "0", "1"]
        m["InitialTransformParameterFileName"] = ["NoInitialTransform"]
        tp2.AddParameterMap(m)
    dummy = itk.image_from_array(np.zeros(shape_zyx, np.float32))
    dummy.SetSpacing((spacing,) * 3)
    dummy.SetOrigin(tuple(float(o) for o in origin))
    with tempfile.TemporaryDirectory() as td:      # transformix otherwise drops deformationField.nii in the cwd
        T = itk.TransformixFilter.New(dummy)
        T.SetTransformParameterObject(tp2)
        T.SetOutputDirectory(td)
        T.ComputeDeformationFieldOn()
        T.LogToConsoleOff()
        T.UpdateLargestPossibleRegion()
        u = np.asarray(T.GetOutputDeformationField()).astype(np.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape_zyx], indexing="ij")
    p = np.stack([xx * spacing + origin[0], yy * spacing + origin[1], zz * spacing + origin[2]], -1)
    return p + u


def map_points(tp, pts_phys):
    """Fixed physical points (N,3 itk order, mm) -> moving physical points via transformix."""
    tp2 = itk.ParameterObject.New()
    for i in range(tp.GetNumberOfParameterMaps()):
        m = tp.GetParameterMap(i)
        m["InitialTransformParameterFileName"] = ["NoInitialTransform"]
        tp2.AddParameterMap(m)
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "pts.txt"
        pf.write_text("point\n%d\n" % len(pts_phys) + "\n".join(" ".join(f"{v:.6f}" for v in p) for p in pts_phys))
        T = itk.TransformixFilter.New(itk.image_from_array(np.zeros((2, 2, 2), np.float32)))
        T.SetTransformParameterObject(tp2)
        T.SetFixedPointSetFileName(str(pf))
        T.SetOutputDirectory(td)
        T.LogToConsoleOff()
        T.UpdateLargestPossibleRegion()
        out = [l.split("OutputPoint = [")[1].split("]")[0].split()
               for l in (Path(td) / "outputpoints.txt").read_text().splitlines()]
    return np.array(out, float)


def outline_mask(smask, dil=8):
    """Where the outline term may be evaluated. The slab's medial/lateral faces are CUTS (atlas brain continues
    beyond them), so along x only the span between each (z,y) column's first and last tissue voxel is used;
    columns without tissue (outside the brain in-plane / natural surface) are used within `dil` vox in-plane."""
    has = smask.any(2)
    nx = smask.shape[2]
    xs = np.arange(nx)
    first = np.where(has, smask.argmax(2), nx)
    last = np.where(has, nx - 1 - smask[:, :, ::-1].argmax(2), -1)
    within = (xs[None, None, :] >= first[..., None]) & (xs[None, None, :] <= last[..., None])
    near = ndi.binary_dilation(has, iterations=dil) & ~has
    return within | near[..., None]


def slab_grid_geom(g, um, shape, n50x):
    """Fixed-image spacing/origin (mm, itk order x,y,z) of a slab grid with voxel size `um`, in the registration frame
    of geometry g (fixed_spacing_mm = 50 um grid spacing incl. start scale; flip along x if g['flip']).
    n50x: number of x voxels of the 50 um grid."""
    fsp = g["fixed_spacing_mm"]
    r = um / 50.0
    o = fsp * (0.5 * r - 0.5)
    ox = o
    if g.get("flip"):
        ox = fsp * ((n50x - 1) - (shape[2] - 0.5) * r + 0.5)
    return fsp * r, (ox, o, o)
