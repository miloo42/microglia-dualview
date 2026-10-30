"""Allen CCFv3 (2017) files: 50 um template/annotation (nrrd), 10 um template/annotation unpacked once to .npy
(memory-mappable), and the region tree. Atlas axes (AP, DV, ML), atlas voxel m at res um sits at CCF position res*m um."""
import json
from functools import cached_property
from pathlib import Path

import numpy as np

ATLAS_DIR = Path(r"X:\Microglia\Atlas\Allen")
MIDLINE_UM = 5700.0      # CCF ML of the midline

MAJOR = {"Isocortex": 315, "OLF": 698, "HPF": 1089, "CTXsp": 703, "STR": 477, "PAL": 803, "TH": 549, "HY": 1097,
         "MB": 313, "P": 771, "MY": 354, "CB": 512, "fiber tracts": 1009, "VS": 73}


class Atlas:
    """Allen files under `atlas_dir`; the 10 um .npy cache goes to `cache_dir` (default <atlas_dir>/npy_cache, 6.8 GB)."""

    def __init__(self, atlas_dir=ATLAS_DIR, cache_dir=None):
        self.dir = Path(atlas_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.dir / "npy_cache"
        self.template50_fn = self.dir / "average_template_50.nrrd"
        self.annot50_fn = self.dir / "annotation" / "ccf_2017" / "annotation_50.nrrd"
        self.template10_fn = self.dir / "average_template_10.nrrd"
        self.annot10_fn = self.dir / "annotation" / "ccf_2017" / "annotation_10.nrrd"
        self.structures_fn = self.dir / "structures_dl" / "structure_graph_1.json"

    def load50(self):
        """(template float32, annotation) at 50 um."""
        import nrrd
        t, _ = nrrd.read(self.template50_fn)
        a, _ = nrrd.read(self.annot50_fn)
        return t.astype(np.float32), a

    def ensure_cache(self, log=print):
        """Unpack the 10 um template and annotation to .npy once (~1 min)."""
        import nrrd
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for name, src in (("template", self.template10_fn), ("annot", self.annot10_fn)):
            dst = self.cache_dir / f"atlas10_{name}.npy"
            if not dst.exists():
                log(f"unpacking {src.name} -> {dst}")
                a, _ = nrrd.read(src)
                np.save(dst, a)
                del a

    def template10(self):
        self.ensure_cache()
        return np.load(self.cache_dir / "atlas10_template.npy", mmap_mode="r")

    def annot10(self):
        self.ensure_cache()
        return np.load(self.cache_dir / "atlas10_annot.npy", mmap_mode="r")

    @cached_property
    def structures(self):
        """id -> dict(acronym, name, color '#rrggbb', parent, parents: ancestor ids incl. self)."""
        root = json.load(open(self.structures_fn))["msg"][0]
        out = {}

        def walk(n, anc):
            a = anc + [n["id"]]
            out[n["id"]] = dict(acronym=n["acronym"], name=n["name"], color="#" + n.get("color_hex_triplet", "AAAAAA"),
                                parent=anc[-1] if anc else None, parents=a)
            for c in n.get("children", []):
                walk(c, a)
        walk(root, [])
        return out

    def descendants(self, sid):
        """sid and every region below it in the tree."""
        return [k for k, v in self.structures.items() if sid in v["parents"]]

    def acronym_to_id(self, acronym):
        return next(k for k, v in self.structures.items() if v["acronym"] == acronym)

    def to_major(self, annot):
        """Map annotation ids to the index (1..len(MAJOR)) of their major parent region; 0 = other/outside."""
        st = self.structures
        lut = {}
        for i in np.unique(annot):
            anc = st.get(int(i), {}).get("parents", [])
            lut[i] = next((k + 1 for k, (_, sid) in enumerate(MAJOR.items()) if sid in anc), 0)
        return np.vectorize(lut.get, otypes=[np.uint8])(annot)
