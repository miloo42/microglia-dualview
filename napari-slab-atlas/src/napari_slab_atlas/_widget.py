"""Slab atlas widget: register the fused zarr shown in napari to the Allen atlas (slab_atlas), show the warped
regions, and turn boxes drawn on the zarr into pos.pos files for follow-up acquisitions (pospos)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from napari.layers import Image, Labels, Shapes
from napari.qt.threading import thread_worker
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
                            QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                            QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from . import pospos

ATLAS_LAYER = "atlas regions"
SELECTED_LAYER = "selected regions"
ROI_LAYER = "ROIs"
# preselected regions (from choosing_ROIs.py)
DEFAULT_REGIONS = ["MBO", "SUM", "MS", "NDB", "LS", "RSP", "ACA", "AD", "AM", "AV", "SUB"]
ORIENTATIONS = {"from orientation test": None, "as acquired": 0, "flipped": 1}


class _Emitter(QObject):
    line = Signal(str)


def _path_row(edit, directory, parent):
    """QLineEdit + browse button."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit)
    b = QPushButton("...")
    b.setFixedWidth(28)

    def browse():
        if directory:
            p = QFileDialog.getExistingDirectory(parent, "Select folder", edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(parent, "Select file", edit.text())
        if p:
            edit.setText(p)
    b.clicked.connect(browse)
    h.addWidget(b)
    return w


class SlabAtlasWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.ws = None
        self.results_dir = None      # None: <zarr name>_atlas next to the zarr
        self.cache_dir = None        # None: default 10 um atlas cache
        self._busy = False
        self._labels = None          # (warped annotation, grid um)
        self._emitter = _Emitter()
        self._emitter.line.connect(self._append_log)

        # everything in a scroll area: the panel is taller than a laptop screen
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        lay = QVBoxLayout(inner)

        # --- slab
        box = QGroupBox("Slab")
        f = QFormLayout(box)
        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._image_changed)
        f.addRow("image layer", self.image_combo)
        self.zarr_edit = QLineEdit()
        f.addRow("zarr", _path_row(self.zarr_edit, True, self))
        self.ch540 = QSpinBox()
        self.ch630 = QSpinBox()
        self.ch630.setValue(1)
        ch = QWidget()
        h = QHBoxLayout(ch)
        h.setContentsMargins(0, 0, 0, 0)
        for lab, sb in (("540/595", self.ch540), ("630/700", self.ch630)):
            h.addWidget(QLabel(lab))
            h.addWidget(sb)
        f.addRow("channels", ch)
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_workspace)
        f.addRow(self.open_btn)
        self.status_lbl = QLabel("no slab opened")
        self.status_lbl.setWordWrap(True)
        f.addRow(self.status_lbl)
        lay.addWidget(box)

        # --- registration
        box = QGroupBox("Registration")
        f = QFormLayout(box)
        self.workers = QSpinBox()
        self.workers.setRange(1, 32)
        self.workers.setValue(5)
        f.addRow("pose-search processes", self.workers)
        self.find_btn = QPushButton("1. Find pose (50 µm, ~20 min)")
        self.find_btn.setToolTip("Preprocessing, 350-start pose search, ranking, orientation test and 50 µm "
                                 "B-spline. Steps with results are skipped.")
        self.find_btn.clicked.connect(self.find_pose)
        f.addRow(self.find_btn)
        self.orient_combo = QComboBox()
        self.orient_combo.addItems(list(ORIENTATIONS))
        orow = QWidget()
        h = QHBoxLayout(orow)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.orient_combo)
        self.orient_btn = QPushButton("Use")
        self.orient_btn.setToolTip("Use the best run of this orientation as the 50 µm result")
        self.orient_btn.clicked.connect(self.choose_orientation)
        h.addWidget(self.orient_btn)
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setToolTip("Figure with the best run of each orientation side by side")
        self.compare_btn.clicked.connect(self.compare_orientations)
        h.addWidget(self.compare_btn)
        f.addRow("orientation", orow)
        self.refine_btn = QPushButton("2. Refine (20 µm vs 10 µm atlas, ~3 min)")
        self.refine_btn.clicked.connect(self.refine)
        f.addRow(self.refine_btn)
        lay.addWidget(box)

        # --- atlas display
        box = QGroupBox("Atlas regions")
        v = QVBoxLayout(box)
        self.show_btn = QPushButton("Show atlas layers")
        self.show_btn.clicked.connect(self.show_atlas)
        v.addWidget(self.show_btn)
        self.search = QLineEdit()
        self.search.setPlaceholderText("search acronym or name")
        self.search.textChanged.connect(self._filter_regions)
        v.addWidget(self.search)
        self.region_list = QListWidget()
        self.region_list.setMinimumHeight(160)
        self.region_list.itemChanged.connect(self._update_selected_layer)
        v.addWidget(self.region_list)
        self.only_checked = QCheckBox("list checked regions only")
        self.only_checked.toggled.connect(self._filter_regions)
        v.addWidget(self.only_checked)
        lay.addWidget(box)

        # --- ROIs -> pos.pos
        box = QGroupBox("ROIs → pos files")
        f = QFormLayout(box)
        srow = QWidget()
        h = QHBoxLayout(srow)
        h.setContentsMargins(0, 0, 0, 0)
        self.shapes_combo = QComboBox()
        h.addWidget(self.shapes_combo)
        b = QPushButton("New")
        b.setToolTip("Add a Shapes layer for drawing ROI rectangles")
        b.clicked.connect(self.new_roi_layer)
        h.addWidget(b)
        f.addRow("shapes layer", srow)
        self.raw_edit = QLineEdit()
        self.raw_edit.setToolTip("Raw acquisition folder with the pos.pos and the .tif files")
        f.addRow("raw folder", _path_row(self.raw_edit, True, self))
        cam = QWidget()
        h = QHBoxLayout(cam)
        h.setContentsMargins(0, 0, 0, 0)
        self.cam_w, self.cam_h = QSpinBox(), QSpinBox()
        for sb, tip in ((self.cam_w, "width"), (self.cam_h, "height")):
            sb.setRange(0, 10000)
            sb.setSpecialValueText("from .tif")
            sb.setToolTip(f"camera {tip} in px; needed when the raw folder has no .tif (only AcqSettings.txt)")
            h.addWidget(sb)
        self.pix_um = QDoubleSpinBox()
        self.pix_um.setDecimals(3)
        self.pix_um.setMinimumWidth(90)
        self.pix_um.setRange(0, 100)
        self.pix_um.setSpecialValueText("from .tif")
        self.pix_um.setToolTip("camera pixel size in the sample (um)")
        h.addWidget(self.pix_um)
        f.addRow("camera w, h (px), µm/px", cam)
        self.camera = QComboBox()
        self.camera.addItems(["from data", "RightCam", "LeftCam"])
        f.addRow("overview camera", self.camera)
        self.scan_mode = QComboBox()
        self.scan_mode.addItems(["from metadata.txt", "X = scan centre", "X = scan start"])
        f.addRow("pos.pos scan mode", self.scan_mode)
        self.overlap = QDoubleSpinBox()
        self.overlap.setRange(0, 0.9)
        self.overlap.setSingleStep(0.05)
        self.overlap.setValue(0.1)
        f.addRow("tile overlap", self.overlap)
        self.single_tile = QCheckBox("one position per ROI (cropped to one FOV)")
        f.addRow(self.single_tile)
        self.views = QComboBox()
        self.views.addItems(["AB", "A", "B"])
        f.addRow("views covering the ROI", self.views)
        self.mfs = QDoubleSpinBox()
        self.mfs.setDecimals(4)
        self.mfs.setRange(0.01, 20)
        self.mfs.setValue(0.5)
        self.mfs.setToolTip("Stage X delta per slice (um), snapped to the nearest value in the MFS file")
        f.addRow("MFS (µm/slice)", self.mfs)
        self.mfs_file = QLineEdit(pospos.MFS_FILE)
        f.addRow("MFS file", _path_row(self.mfs_file, False, self))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0, 5000)
        self.margin.setSuffix(" µm")
        f.addRow("margin", self.margin)
        self.pos_out = QLineEdit()
        self.pos_out.setPlaceholderText("default: <results folder>/ROIs_pos")
        f.addRow("output folder", _path_row(self.pos_out, True, self))
        self.pos_btn = QPushButton("Write pos files")
        self.pos_btn.clicked.connect(self.write_pos)
        f.addRow(self.pos_btn)
        lay.addWidget(box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        lay.addWidget(self.log)

        self.viewer.layers.events.inserted.connect(self._refresh_combos)
        self.viewer.layers.events.removed.connect(self._refresh_combos)
        self._refresh_combos()

    # ----------------------------------------------------------------------------------------------------------
    # helpers

    def _append_log(self, s):
        self.log.appendPlainText(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _log(self, *a):
        self._emitter.line.emit(" ".join(str(x) for x in a))

    def _refresh_combos(self, *_):
        for combo, kind in ((self.image_combo, Image), (self.shapes_combo, Shapes)):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            names = [l.name for l in self.viewer.layers if isinstance(l, kind)]
            combo.addItems(names)
            if cur in names:
                combo.setCurrentText(cur)
            combo.blockSignals(False)
        if not self.zarr_edit.text():
            self._image_changed()

    def _image_layer(self):
        name = self.image_combo.currentText()
        return self.viewer.layers[name] if name in self.viewer.layers else None

    def _image_changed(self, *_):
        layer = self._image_layer()
        p = layer.source.path if layer is not None else None
        if p:
            p = Path(p)
            while p.suffix != ".zarr" and p.parent != p:      # a pyramid level or channel path inside the zarr
                p = p.parent
            if p.suffix == ".zarr" and str(p) != self.zarr_edit.text():
                self.zarr_edit.setText(str(p))
                self.open_workspace()

    def _set_busy(self, busy):
        self._busy = busy
        for b in (self.open_btn, self.find_btn, self.orient_btn, self.compare_btn, self.refine_btn, self.pos_btn):
            b.setEnabled(not busy)

    def _run(self, fn, done=None):
        """Run fn() in a thread; done(result) on success, in the GUI thread."""
        if self._busy:
            return
        if self.ws is None and not self.open_workspace():
            return
        self._set_busy(True)
        worker = thread_worker(fn)()

        def finished(res):
            if done is not None:
                done(res)
            self._update_status()

        worker.returned.connect(finished)
        worker.errored.connect(lambda e: self._log(f"ERROR: {type(e).__name__}: {e}"))
        worker.finished.connect(lambda: self._set_busy(False))
        worker.start()

    def _update_status(self):
        if self.ws is None:
            return
        st = self.ws.status()
        o = self.ws.orientation()
        txt = [f"results: {self.ws.out}"]
        txt.append("done: " + (", ".join(k for k, v in st.items() if v) or "nothing yet"))
        if o:
            txt.append(f"orientation test: {o['verdict']}")
        if st["reg50"]:
            g = self.ws.geometry(fine=False)
            txt.append("50 µm result: rank %s, %s, medial face %.0f µm from midline, yaw %.1f°, roll %.1f°" % (
                g.get("rank", "?"), "flipped" if g["flip"] else "as acquired", g["pose"]["face_ml_um"],
                g["pose"]["yaw"], g["pose"]["roll"]))
        self.status_lbl.setText("\n".join(txt))

    # ----------------------------------------------------------------------------------------------------------
    # slab / registration

    def open_workspace(self):
        from slab_atlas import Workspace
        z = self.zarr_edit.text().strip()
        if not z or not Path(z).exists():
            self._log("select the fused .zarr first")
            return False
        try:
            self.ws = Workspace(z, self.results_dir, ch_540=self.ch540.value(), ch_630=self.ch630.value(),
                                cache_dir=self.cache_dir)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._log(f"cannot open {z}: {e}")
            self.ws = None
            return False
        self._log(f"opened {z} ({self.ws.slab.l0_um:.4f} µm voxels, shape {self.ws.slab.l0_shape})")
        self._update_status()
        if self.ws.labels()[0] is not None:
            self.show_atlas()
        return True

    def find_pose(self):
        from slab_atlas import pipeline
        ws, workers, log = self.ws, self.workers.value(), self._log

        def done(verdict):
            if verdict == "undecided":
                self._log("orientation undecided: press 'Compare', then pick an orientation and press 'Use'")
            else:
                self.show_atlas()
        if ws is None and not self.open_workspace():
            return
        ws = self.ws
        self._run(lambda: pipeline.find_pose(ws, workers=workers, log=log), done)

    def choose_orientation(self):
        from slab_atlas import pipeline
        flip, ws, log = ORIENTATIONS[self.orient_combo.currentText()], self.ws, self._log
        if ws is None:
            return
        self._run(lambda: pipeline.choose_pose(ws, flip, log=log), lambda _: self.show_atlas())

    def compare_orientations(self):
        from slab_atlas import qc
        ws, log = self.ws, self._log
        if ws is None:
            return
        self._run(lambda: qc.compare_50um(ws, log=log), lambda fn: os.startfile(fn))

    def refine(self):
        from slab_atlas import pipeline
        ws, log = self.ws, self._log
        if ws is None:
            return
        if not ws.status()["reg50"]:
            self._log("run 'Find pose' (and choose an orientation) first")
            return
        self._run(lambda: pipeline.refine_20um(ws, log=log), lambda _: self.show_atlas())

    # ----------------------------------------------------------------------------------------------------------
    # atlas layers

    def _placement(self, um):
        """scale/translate of a slab grid with voxel `um` in the world frame of the image layer."""
        img = self._image_layer()
        sc = np.asarray(img.scale[-3:], float) if img is not None else np.ones(3)
        tr = np.asarray(img.translate[-3:], float) if img is not None else np.zeros(3)
        f = um / self.ws.slab.l0_um
        return sc * f, tr + sc * (0.5 * f - 0.5)

    def show_atlas(self):
        if self.ws is None:
            return
        wa, um = self.ws.labels()
        if wa is None:
            self._log("no registration yet")
            return
        self._labels = (wa, um)
        st = self.ws.atlas.structures
        present = [int(i) for i in np.unique(wa) if i]
        colors = {None: np.zeros(4), 0: np.zeros(4)}
        colors.update({i: st[i]["color"] for i in present if i in st})
        features = pd.DataFrame(dict(index=[0] + present,
                                     acronym=["-"] + [st.get(i, {}).get("acronym", "?") for i in present],
                                     name=["outside"] + [st.get(i, {}).get("name", "?") for i in present]))
        scale, translate = self._placement(um)
        layer = self._labels_layer(ATLAS_LAYER, wa, scale, translate, colors)
        layer.features = features
        layer.opacity = 0.35
        self._fill_region_list(present)
        self._update_selected_layer()
        self._log(f"atlas regions at {um:.0f} µm ({len(present)} regions in the slab)")

    def _labels_layer(self, name, data, scale, translate, colors):
        cmap = DirectLabelColormap(color_dict=colors)
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            layer = self.viewer.layers[name]
            layer.data = data
            layer.scale, layer.translate = scale, translate
            layer.colormap = cmap
        else:
            layer = self.viewer.add_labels(data, name=name, scale=scale, translate=translate, colormap=cmap)
            layer.editable = False
        return layer

    def _fill_region_list(self, present):
        """Regions in the slab and their ancestors, in tree order, indented by depth."""
        st = self.ws.atlas.structures
        checked = self._checked() or {self.ws.atlas.acronym_to_id(a) for a in DEFAULT_REGIONS}
        show = set()
        for i in present:
            show.update(st.get(i, {}).get("parents", []))
        self.region_list.blockSignals(True)
        self.region_list.clear()
        for sid in st:                                  # dict order = depth-first tree order
            if sid not in show:
                continue
            s = st[sid]
            it = QListWidgetItem("  " * (len(s["parents"]) - 1) + f"{s['acronym']}  {s['name']}")
            it.setData(Qt.UserRole, sid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if sid in checked else Qt.Unchecked)
            self.region_list.addItem(it)
        self.region_list.blockSignals(False)
        self._filter_regions()

    def _checked(self):
        return {self.region_list.item(i).data(Qt.UserRole) for i in range(self.region_list.count())
                if self.region_list.item(i).checkState() == Qt.Checked}

    def _filter_regions(self, *_):
        q = self.search.text().strip().lower()
        only = self.only_checked.isChecked()
        for i in range(self.region_list.count()):
            it = self.region_list.item(i)
            hide = (q and q not in it.text().lower()) or (only and it.checkState() != Qt.Checked)
            it.setHidden(bool(hide))

    def _update_selected_layer(self, *_):
        if self._labels is None:
            return
        wa, um = self._labels
        st = self.ws.atlas.structures
        sel = sorted(self._checked(), key=lambda s: len(st[s]["parents"]))   # children drawn over parents
        out = np.zeros(wa.shape, np.uint32)
        for sid in sel:
            out[np.isin(wa, self.ws.atlas.descendants(sid))] = sid
        colors = {None: np.zeros(4), 0: np.zeros(4)}
        colors.update({s: st[s]["color"] for s in sel})
        scale, translate = self._placement(um)
        layer = self._labels_layer(SELECTED_LAYER, out, scale, translate, colors)
        layer.features = pd.DataFrame(dict(index=[0] + sel, acronym=["-"] + [st[s]["acronym"] for s in sel],
                                           name=["-"] + [st[s]["name"] for s in sel]))
        if self.only_checked.isChecked():
            self._filter_regions()

    # ----------------------------------------------------------------------------------------------------------
    # ROIs -> pos.pos

    def new_roi_layer(self):
        img = self._image_layer()
        ndim = img.ndim if img is not None else 3
        layer = self.viewer.add_shapes(name=ROI_LAYER, ndim=ndim, edge_color="yellow", face_color="transparent",
                                       edge_width=3)
        layer.mode = "add_rectangle"
        self.shapes_combo.setCurrentText(layer.name)

    def write_pos(self):
        img = self._image_layer()
        name = self.shapes_combo.currentText()
        if img is None or name not in self.viewer.layers or not self.viewer.layers[name].data:
            self._log("draw ROI rectangles on a Shapes layer first")
            return
        if self.ws is None and not self.open_workspace():
            return
        shapes_layer = self.viewer.layers[name]
        # shape vertices -> world -> level-0 voxels of the image
        shapes = []
        for s in shapes_layer.data:
            pts = []
            for v in s:
                w = np.asarray(shapes_layer.data_to_world(v), float)
                w = np.concatenate([np.zeros(max(img.ndim - len(w), 0)), w])[-img.ndim:]
                pts.append(np.asarray(img.world_to_data(w), float)[-3:])
            shapes.append(np.array(pts))
        out = self.pos_out.text().strip() or str(self.ws.out / "ROIs_pos")
        raw = self.raw_edit.text().strip()
        if not raw:
            self._log("select the raw acquisition folder")
            return
        kw = dict(overlap=self.overlap.value(), single_tile=self.single_tile.isChecked(), views=self.views.currentText(),
                  mfs=self.mfs.value(), mfs_file=self.mfs_file.text().strip(), margin=self.margin.value(),
                  cam_width=self.cam_w.value() or None, cam_height=self.cam_h.value() or None,
                  pix_um=self.pix_um.value() or None,
                  camera=None if self.camera.currentIndex() == 0 else self.camera.currentText(),
                  scan_from_start=[None, False, True][self.scan_mode.currentIndex()],
                  source=f"napari layer '{name}' ({out}/ROIs_shapes.csv)")
        zarr_fn, log = str(self.ws.slab.path), self._log

        def job():
            os.makedirs(out, exist_ok=True)
            shapes_layer.save(os.path.join(out, "ROIs_shapes.csv"))
            return pospos.write_pos_files(zarr_fn, raw, shapes, out, log=log, **kw)
        self._run(job, lambda rois: self._log(f"{len(rois)} ROIs written to {out}"))
