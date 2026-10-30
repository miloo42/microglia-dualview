"""Chunked GPU Richardson-Lucy deconvolution of the two fused dual-view stacks.

Each view is deconvolved with its own (rotated) theoretical PSF, generated in
theoretical-psf/theo_lightsheet.ipynb. Chunks are processed with dask
map_overlap so the PSF edge effects are trimmed away.
"""
import gc
import time
import traceback
from pathlib import Path

import cupy as cp
import dask
import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster
from scipy import ndimage as ndi
from tifffile import imread
from tnia.deconvolution.richardson_lucy import richardson_lucy_cp

# ---------------------------------------------------------------- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PSF_DIR = SCRIPT_DIR / "theoretical-psf"

DATA_DIR = Path(R"X:\Microglia\2026_10_24_S1")
FUSED_DIR = DATA_DIR / "2026_10_19_s1_cx_dw_630_700+540_595_MFS05_1x1_AB_100ms_views_crop.zarr"

# view name -> (input zarr, psf file, raw copy zarr, deconvolved zarr)
VIEWS = {
    "A": (FUSED_DIR / "fused_tp_0_vs_2.zarr" / "0",
          PSF_DIR / "psfr1.tiff",
          DATA_DIR / "viewA.zarr",
          DATA_DIR / "viewA_decon.zarr"),
    "B": (FUSED_DIR / "fused_tp_0_vs_3.zarr" / "0",
          PSF_DIR / "psfr2.tiff",
          DATA_DIR / "viewB.zarr",
          DATA_DIR / "viewB_decon.zarr"),
}

# ----------------------------------------------------------- parameters ----
ITERATIONS = 50
# chunk edge = CHUNK_FACTOR * psf size; 3 x 75 px fits on the GPU
# (300x375x375 seemed to be the max for this implementation)
CHUNK_FACTOR = 3
SAVE_RAW_COPY = False  # also write the undeconvolved input next to the output
GPU_ID = 0  # with multiple GPUs, use the queue logic from the original repo
            # and do not use more workers than GPUs
# RL fills empty input (outside a view's coverage) with bright extrapolated
# values; these are zeroed in the output, grown by this margin (voxels) since
# the artefact reaches a few voxels past the empty region
EMPTY_MARGIN = 5


def free_gpu_memory():
    """Release CuPy's cached device/pinned memory and FFT plans.

    CuPy keeps freed blocks and cuFFT plans (one per distinct chunk shape) around
    for reuse; over many chunks this shows up as ever-growing unmanaged memory.
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()


def rl_dask_task(img_chunk, psf, num_iterations, block_id=None):
    """Deconvolve one chunk on the GPU; chunks with an empty mask are skipped."""
    # mask computed here instead of as a second dask array: halves the data
    # dask keeps in flight for the overlap
    mask_chunk = img_chunk > 0
    if not mask_chunk.any():
        print(f"block {block_id}: empty, skipping")
        return np.zeros(img_chunk.shape, dtype=np.float32)

    print(f"block {block_id}: deconvolving")
    try:
        cp.cuda.Device(GPU_ID).use()

        result = richardson_lucy_cp(img_chunk.astype(np.float32), psf,
                                    num_iterations, noncirc=True,
                                    mask=mask_chunk)
        result = np.asarray(result, dtype=np.float32)
        # the overlap (psf size) is wider than the margin, so this is exact
        # at chunk edges
        if not mask_chunk.all():
            empty = ndi.binary_dilation(~mask_chunk, iterations=EMPTY_MARGIN)
            result[empty] = 0
        return result
    except Exception:
        traceback.print_exc()
        raise
    finally:
        del mask_chunk
        gc.collect()
        free_gpu_memory()


def load_psf(path):
    psf = imread(path).astype(np.float32)
    return psf / psf.sum()


def deconvolve_view(in_path, psf_path, raw_path, out_path):
    img = da.from_zarr(in_path)
    psf = load_psf(psf_path)

    print("Image shape ", img.shape)

    if SAVE_RAW_COPY:
        da.to_zarr(img, raw_path, overwrite=True)

    overlap = psf.shape[0]
    chunk_size = (overlap * CHUNK_FACTOR,) * 3

    img = img.rechunk(chunks=chunk_size)
    out = da.map_overlap(rl_dask_task, img, depth=(overlap,) * 3,
                         dtype=np.float32, psf=psf,
                         num_iterations=ITERATIONS)

    start_time = time.time()
    da.to_zarr(out, out_path, overwrite=True)
    print(f"done in {time.time() - start_time:.0f} s -> {out_path}")


if __name__ == "__main__":
    # don't let the scheduler load many input chunks ahead of the (slow) GPU
    dask.config.set({"distributed.scheduler.worker-saturation": 1.0})
    cluster = LocalCluster(
        n_workers=1,
        memory_limit="10GB",  # per worker
        # separate worker process: its memory is measured on its own (not the
        # client's), and the nanny restarts it instead of hanging if it fills up
        processes=True,
        threads_per_worker=1,  # one chunk on the GPU at a time
        dashboard_address="localhost:8787",
    )
    client = Client(cluster)
    print(client.dashboard_link)
    dev_id = cp.cuda.Device(GPU_ID).id
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    name = props["name"].decode()
    print(f"Current GPU: {name}")

    for name, paths in VIEWS.items():
        print(f"Starting view {name}")
        deconvolve_view(*paths)
