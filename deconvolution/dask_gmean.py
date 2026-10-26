import dask.array as da
import matplotlib.pyplot as plt
from scipy.stats import gmean
import numpy as np


import time
import dask

import traceback
from dask.distributed import Client, LocalCluster


def dask_gmean(a,b):
    return gmean(np.stack([a, b]),axis = 0)


if __name__ == '__main__':
    print('yay')
    bytes_per_gb = 1024*1024*1024
    debug = True

    cluster = LocalCluster(
    n_workers=1,  # Number of workers
    memory_limit="16GB", # per worker
    processes = False,
    threads_per_worker = 12,
    dashboard_address = 'localhost:8787',
    )

    c = Client(cluster)

    dataA = da.from_zarr( r"X:\Microglia\2025_11_06_M1_dw\viewAbigcrop_decon.zarr")
    dataB = da.from_zarr( r"X:\Microglia\2025_11_06_M1_dw\viewBbigcrop_decon.zarr")
    outimgpath = r"X:\Microglia\2025_11_06_M1_dw\big_swdecon_fused.zarr"

    out = da.map_blocks(dask_gmean, dataA, dataB)

    out.to_zarr(outimgpath, overwrite = True)
    print('Done')

    