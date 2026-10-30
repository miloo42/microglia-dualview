"""Fuse the two deconvolved views by voxel-wise geometric mean."""
from pathlib import Path

import dask.array as da
from dask.distributed import Client, LocalCluster

# ---------------------------------------------------------------- paths ----
DATA_DIR = Path(r"X:\Microglia\2025_11_06_M1_dw")
VIEW_A_PATH = DATA_DIR / "viewAbigcrop_decon.zarr"
VIEW_B_PATH = DATA_DIR / "viewBbigcrop_decon.zarr"
OUT_PATH = DATA_DIR / "big_swdecon_fused.zarr"


if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=1,
        memory_limit="16GB",  # per worker
        processes=False,
        threads_per_worker=12,
        dashboard_address="localhost:8787",
    )
    client = Client(cluster)
    print(client.dashboard_link)

    view_a = da.from_zarr(VIEW_A_PATH)
    view_b = da.from_zarr(VIEW_B_PATH)

    # geometric mean of two arrays; same as scipy gmean but without the
    # log(0) warnings and the stack copy
    fused = da.sqrt(view_a * view_b)

    fused.to_zarr(OUT_PATH, overwrite=True)
    print(f"Done -> {OUT_PATH}")
