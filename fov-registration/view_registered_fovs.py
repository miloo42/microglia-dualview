"""Show the registered dualview FOVs in the lowres overview in napari.

Reads the txt written by fov_registration.register_fovs. Everything is shown in
lowres micrometers: the lowres pyramid with its voxel size, each FOV at full
resolution with its own voxel size and the registered translation.

Usage (napari-env):
    python view_registered_fovs.py <path to fov_registration.txt>
"""
import sys
from pathlib import Path

import dask.array as da
import napari
import numpy as np

from fov_registration import read_registration

# downsampling steps of the lazy FOV pyramid (napari uses the coarsest one in 3D)
FOV_PYRAMID_FACTORS = (1, 2, 4, 8)


def lowres_pyramid(zarr_path, channel):
    levels = sorted((p for p in Path(zarr_path).iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    return [da.from_zarr(str(p))[0, channel] for p in levels]


def fov_pyramid(fov):
    return [fov if f == 1 else da.coarsen(np.mean, fov, {0: f, 1: f, 2: f}, trim_excess=True).astype(fov.dtype)
            for f in FOV_PYRAMID_FACTORS]


def main(txt_path):
    header, rows = read_registration(txt_path)
    lowres_voxel = float(header['lowres_voxel_um'])
    n_channels = da.from_zarr(str(Path(header['lowres_zarr']) / '0')).shape[1]

    viewer = napari.Viewer(ndisplay=3)
    for c in range(n_channels):
        viewer.add_image(lowres_pyramid(header['lowres_zarr'], c), multiscale=True,
                         scale=(lowres_voxel,) * 3, name=f'lowres c{c}',
                         blending='additive', visible=c == int(header['lowres_channel']))

    for row in rows:
        voxel = float(row['voxel_um'])
        translate = [float(row[k]) for k in ('tz', 'ty', 'tx')]
        viewer.add_image(fov_pyramid(da.from_zarr(row['path'])), multiscale=True,
                         scale=(voxel,) * 3, translate=translate, name=row['name'],
                         blending='additive', colormap='magenta')
        print(row['name'], 'translate_um', translate, 'pearson r', row['pearson_r'])

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'
    napari.run()


if __name__ == '__main__':
    main(sys.argv[1])
