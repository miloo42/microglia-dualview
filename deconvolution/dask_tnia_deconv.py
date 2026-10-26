"""Chunked GPU Richardson-Lucy deconvolution of the two fused dual-view stacks.

Each view is deconvolved with its own (rotated) theoretical PSF, generated in
theoretical-psf/theo_lightsheet.ipynb. Chunks are processed with dask
map_overlap so the PSF edge effects are trimmed away.
"""
import time
import traceback
from pathlib import Path

import cupy as cp
import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster
from tifffile import imread
from tnia.deconvolution.richardson_lucy import richardson_lucy_cp

# ---------------------------------------------------------------- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PSF_DIR = SCRIPT_DIR / "theoretical-psf"

DATA_DIR = Path(r"X:\Microglia\2025_11_06_M1_dw")
FUSED_DIR = DATA_DIR / "22025_11_06_M1_dw_640_700_MFS05_1x1_50ms_1_views"

# view name -> (input zarr, psf file, raw copy zarr, deconvolved zarr)
VIEWS = {
    "A": (FUSED_DIR / "fused_tp_0_vs_0.zarr" / "0",
          PSF_DIR / "psfr1.tiff",
          DATA_DIR / "viewAbigcrop.zarr",
          DATA_DIR / "viewAbigcrop_decon.zarr"),
    "B": (FUSED_DIR / "fused_tp_0_vs_1.zarr" / "0",
          PSF_DIR / "psfr2.tiff",
          DATA_DIR / "viewBbigcrop.zarr",
          DATA_DIR / "viewBbigcrop_decon.zarr"),
}

# ----------------------------------------------------------- parameters ----
ITERATIONS = 50
# chunk edge = CHUNK_FACTOR * psf size; 3 x 75 px fits on the GPU
# (300x375x375 seemed to be the max for this implementation)
CHUNK_FACTOR = 3
SAVE_RAW_COPY = True  # also write the undeconvolved input next to the output
GPU_ID = 0  # with multiple GPUs, use the queue logic from the original repo
            # and do not use more workers than GPUs


def rl_dask_task(img_chunk, mask_chunk, psf, num_iterations, block_id=None):
    """Deconvolve one chunk on the GPU; chunks with an empty mask are skipped."""
    if not mask_chunk.any():
        print(f"block {block_id}: empty, skipping")
        return np.zeros(img_chunk.shape, dtype=np.float32)

    print(f"block {block_id}: deconvolving")
    try:
        cp.cuda.Device(GPU_ID).use()
        result = richardson_lucy_cp(img_chunk, psf, num_iterations,
                                    noncirc=True, mask=mask_chunk)
        return cp.asnumpy(result).astype(np.float32, copy=False)
    except Exception:
        traceback.print_exc()
        raise


def load_psf(path):
    psf = imread(path).astype(np.float32)
    return psf / psf.sum()


def deconvolve_view(in_path, psf_path, raw_path, out_path):
    img = da.from_zarr(in_path)
    psf = load_psf(psf_path)

    if SAVE_RAW_COPY:
        da.to_zarr(img, raw_path, overwrite=True)

    overlap = psf.shape[0]
    chunk_size = (overlap * CHUNK_FACTOR,) * 3

    img = img.rechunk(chunks=chunk_size)
    mask = img > 0
    out = da.map_overlap(rl_dask_task, img, mask, depth=(overlap,) * 3,
                         dtype=np.float32, psf=psf,
                         num_iterations=ITERATIONS)

    start_time = time.time()
    da.to_zarr(out, out_path, overwrite=True)
    print(f"done in {time.time() - start_time:.0f} s -> {out_path}")


if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=1,
        memory_limit="10GB",  # per worker
        processes=False,
        threads_per_worker=1,  # one chunk on the GPU at a time
        dashboard_address="localhost:8787",
    )
    client = Client(cluster)
    print(client.dashboard_link)

    for name, paths in VIEWS.items():
        print(f"Starting view {name}")
        deconvolve_view(*paths)
