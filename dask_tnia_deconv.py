import cupy as cp
import numpy as np
import os
from tifffile import imread, imwrite
import dask.array as da
import matplotlib.pyplot as plt

from tnia.deconvolution.richardson_lucy import richardson_lucy_cp

import time
import dask

import traceback
from dask.distributed import Client, LocalCluster


def rl_dask_task(img_chunk, mask_chunk, psf, numiterations, block_info=None, block_id=None, thread_id=None):
    device_num=0
    # with multiple gpus, use the queue logic from the original repo
    # do not use more workers than gpus

    print('block id is', block_id)
    #print('block info is', block_info)
    try:
        if np.max(mask_chunk) == 0:
            print('Skipping empty chunk')
            return cp.zeros_like(img_chunk)
            
        cp.cuda.Device(device_num).use()
        result = richardson_lucy_cp(img_chunk, psf, numiterations, noncirc = True, mask=mask_chunk)                
        return result
    
    except Exception as e:
        traceback.print_exc()
        print()
        print("EXCEPTION",e)
        pass


if __name__ == '__main__':
    print('yay')
    bytes_per_gb = 1024*1024*1024
    debug = True

    cluster = LocalCluster(
    n_workers=1,  # Number of workers
    memory_limit="10GB", # per worker
    processes = False,
    threads_per_worker = 1,
    dashboard_address = 'localhost:8787',
    )

    c = Client(cluster)

    data = da.from_zarr(r"X:\Microglia\2025_11_06_M1_dw\22025_11_06_M1_dw_640_700_MFS05_1x1_50ms_1_views\fused_tp_0_vs_0.zarr\0")

    path_psf = r"X:\Microglia\theoreticalpsf_singledeconv"
    psf_a_filename = 'psfr1.tiff'
    psf = imread(os.path.join(path_psf, psf_a_filename)).astype(np.float32)
    psf = psf/psf.sum()

    img = data
    outimgpath = r"X:\Microglia\2025_11_06_M1_dw\viewAbigcrop.zarr"
    da.to_zarr(img,outimgpath, overwrite = True)

    ########### parameters ################
    overlap = psf.shape[0]
    depth = (overlap,overlap,overlap)
    chunk_size = (psf.shape[0]*3, psf.shape[0]*3, psf.shape[0]*3) # with this specific implementation, 300x375x375 seems to be the biggest we can fit on a gpu - thats ok!

    iterations = 50


    # gpu_mem_ = gpu_mem()/bytes_per_gb
    # rl_mem_ = rl_mem_footprint(img, psf, depth=depth)/bytes_per_gb


    start_time = time.time()

    daskimg = img.rechunk(chunks=chunk_size)
    daskmask = daskimg>0

    out = dask.array.map_overlap(rl_dask_task, daskimg, daskmask, depth=(overlap,overlap,overlap), dtype=np.float32, psf=psf, numiterations=iterations)

    #out_img = out.compute(num_workers=1)

    outpath =r"X:\Microglia\2025_11_06_M1_dw\viewAbigcrop_decon.zarr"
    print('Starting process')
    da.to_zarr(out,outpath, overwrite = True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)

    print('Starting with B')
    data = da.from_zarr(r"X:\Microglia\2025_11_06_M1_dw\22025_11_06_M1_dw_640_700_MFS05_1x1_50ms_1_views\fused_tp_0_vs_1.zarr\0")
    path_psf = r"X:\Microglia\theoreticalpsf_singledeconv"
    psf_b_filename = 'psfr2.tiff'
    outimgpath = r"X:\Microglia\2025_11_06_M1_dw\viewBbigcrop.zarr"
    outpath =r"X:\Microglia\2025_11_06_M1_dw\viewBbigcrop_decon.zarr"

    psf = imread(os.path.join(path_psf, psf_b_filename)).astype(np.float32)
    psf = psf/psf.sum()

    img = data
    da.to_zarr(img,outimgpath, overwrite = True)


    overlap = psf.shape[0]
    depth = (overlap,overlap,overlap)
    chunk_size = (psf.shape[0]*3, psf.shape[0]*3, psf.shape[0]*3) # with this specific implementation, 300x375x375 seems to be the biggest we can fit on a gpu - thats ok!
    # gpu_mem_ = gpu_mem()/bytes_per_gb
    # rl_mem_ = rl_mem_footprint(img, psf, depth=depth)/bytes_per_gb
    iterations = 50

    start_time = time.time()

    daskimg = img.rechunk(chunks=chunk_size)
    daskmask = daskimg>0

    out = dask.array.map_overlap(rl_dask_task, daskimg, daskmask, depth=(overlap,overlap,overlap), dtype=np.float32, psf=psf, numiterations=iterations)

    #out_img = out.compute(num_workers=1)

    print('Starting process')
    da.to_zarr(out,outpath, overwrite = True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)