import os

import apoc
import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask.distributed import LocalCluster
from dask_image import ndmeasure, ndmorph
from skimage.measure import regionprops
from apoc import PixelClassifier
import pyopencl as cl
import tifffile

dask.config.set({'temporary_directory': r"X:\dask_tmp"})

cluster = LocalCluster( n_workers=1,
                        memory_limit="10GB", 
                        threads_per_worker = 1,
                        processes = False, 
                        host = 'localhost', 
                        scheduler_port='8688', 
                        dashboard_address=':1415',
                        ) 

client = cluster.get_client()


print(client)
print(client.dashboard_link)



def segment(chunk,clf):
    if chunk.max() < 500: # very important to avoid the empty space
        return np.zeros_like(chunk).astype(bool)
    
    try:
        seg = np.asarray(clf.predict(image=chunk), dtype = int)
        seg[seg>1] = 0
        
        return seg
    except cl.LogicError as e:
        print(f"OpenCL Error: {e}")
        print(f"Error code: {e.code}")
        print(f"Error routine: {e.routine}")




folder = r"X:\Microglia\2025_11_06_M1_dw"

out = r"X:\Microglia\2025_11_06_M1_dw\segmentation\outbig"
filenames = ["2025_11_06_M1_dw_1_swdf.zarr"]
imgnames = ['big']
                
ims = [da.from_zarr(os.path.join(folder,file)) for file in filenames]

clf = PixelClassifier(opencl_filename=r"X:\Microglia\2025_11_06_M1_dw\segmentation\PixelClassifier2.cl")

print('Processing...')
for i in range(len(imgnames)):
    print(i)
    segm = da.map_blocks(segment, ims[i], clf = clf, meta=np.array((), dtype=int))

    da.to_zarr(segm.astype(bool), os.path.join(out,"label" + str(imgnames[i]) + "_bin.zarr"), overwrite = True)