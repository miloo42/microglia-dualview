"""Registration of a light-sheet sagittal brain slab (fused OME-Zarr) to the Allen Mouse Brain Atlas (CCFv3)."""
from .atlas import Atlas, MAJOR
from .slab import Slab
from .pipeline import (Workspace, prepare, pose_search, rank_poses, refine_50um, orientation_test, choose_pose,
                       refine_20um, find_pose, run_all)
