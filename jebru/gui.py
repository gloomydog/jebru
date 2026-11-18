"""jebru GUI (PySide6 + pyqtgraph)."""
from __future__ import annotations

import copy
import csv
import functools
import inspect
import os
import sys
import traceback

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import io, processing as P, analysis as A

pg.setConfigOptions(antialias=True, background="w", foreground="k")

MODES = ("zoom", "integrate", "peak", "baseline", "reference", "phase")
MODE_LABEL = {"zoom": "Zoom", "integrate": "Integrate", "peak": "Peak", "baseline": "Baseline", "reference": "Reference", "phase": "Phase pivot"}
REGION_BRUSH = (30, 120, 220, 35)
REGION_BRUSH_OFF = (150, 150, 150, 30)
UNDO_LIMIT = 200


def undoable(fn):
    """Record an undo snapshot before the (outermost) user action runs.

    Also drops extra positional args that Qt signals pass (e.g. `checked`)."""
    params = list(inspect.signature(fn).parameters.values())[1:]
    var = any(p.kind == p.VAR_POSITIONAL for p in params)
    npos = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)

    @functools.wraps(fn)
    def wrapper(self, *args, **kw):
        if not var:
            args = args[:npos]
        if self._restoring or self.pr is None:
            return fn(self, *args, **kw)
        outer = self._undo_depth == 0
        snap = None
        if outer:
            self._flush_view()
            snap = self._snapshot()
        self._undo_depth += 1
        try:
            return fn(self, *args, **kw)
        finally:
            self._undo_depth -= 1
            if outer:
                if not self._same_state(snap, self._snapshot()):
                    self._push_undo(snap)
                self._settled_view = self._view()
    return wrapper


# ======================================================================
class SpectrumViewBox(pg.ViewBox):
    """ViewBox that routes left-drag / click according to the current mode."""
    sigDragRegion = QtCore.Signal(float, float, bool)   # x0, x1, finished
    sigClick = QtCore.Signal(float, float)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mode = "zoom"
        self.setMouseMode(pg.ViewBox.RectMode)
        self._drag0 = None

    def mouseDragEvent(self, ev, axis=None):
        if self.mode in ("integrate", "baseline") and ev.button() == Qt.LeftButton:
            ev.accept()
            p = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self._drag0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
            if self._drag0 is not None:
                self.sigDragRegion.emit(self._drag0, p.x(), ev.isFinish())
            if ev.isFinish():
                self._drag0 = None
            return
        if ev.button() == Qt.MiddleButton:
            # middle drag = pan
            self.setMouseMode(pg.ViewBox.PanMode)
            super().mouseDragEvent(ev, axis)
            if ev.isFinish():
                self.setMouseMode(pg.ViewBox.RectMode)
            return
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.mode in ("peak", "baseline", "reference", "phase"):
            ev.accept()
            p = self.mapSceneToView(ev.scenePos())
            self.sigClick.emit(p.x(), p.y())
            return
        if ev.button() == Qt.LeftButton and ev.double():
            ev.accept()
            self.autoRange()
            return
        if ev.button() == Qt.RightButton:
            ev.accept()
            self.scaleBy(x=1.6, center=self.mapSceneToView(ev.scenePos()))
            return
        super().mouseClickEvent(ev)

    def wheelEvent(self, ev, axis=None):
        # wheel = vertical scale (intensity), Ctrl+wheel = horizontal zoom
        mods = ev.modifiers()
        d = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        s = 0.85 ** (d / 120.0)
        if mods & Qt.ControlModifier:
            c = self.mapSceneToView(ev.scenePos())
            self.scaleBy(x=s, center=c)
        else:
            yr = self.viewRange()[1]
            lo = yr[0]
            self.setYRange(lo * s, lo + (yr[1] - lo) * s, padding=0)
        ev.accept()


# ======================================================================
class PhaseDialog(QtWidgets.QDialog):
    changed = QtCore.Signal(float, float)

    def __init__(self, p0, p1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual phase correction")
        lay = QtWidgets.QFormLayout(self)
        self.s0 = self._row(lay, "Zero order p0 (°)", p0, 180)
        self.s1 = self._row(lay, "First order p1 (°)", p1, 720)
        lay.addRow(QtWidgets.QLabel("Set the pivot by clicking the spectrum in 'Phase pivot' mode"))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def _row(self, lay, label, val, rng):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        sl = QtWidgets.QSlider(Qt.Horizontal)
        sl.setRange(int(-rng * 10), int(rng * 10))
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(-rng * 4, rng * 4)
        sp.setDecimals(1)
        sp.setSingleStep(0.5)
        sp.setValue(val)
        sl.setValue(int(np.clip(val, -rng, rng) * 10))
        sl.valueChanged.connect(lambda v: sp.setValue(v / 10))
        sp.valueChanged.connect(lambda v: (sl.blockSignals(True), sl.setValue(int(np.clip(v, -rng, rng) * 10)),
                                           sl.blockSignals(False), self._emit()))
        sl.setMinimumWidth(320)
        h.addWidget(sl)
        h.addWidget(sp)
        lay.addRow(label, w)
        return sp

    def _emit(self):
        self.changed.emit(self.s0.value(), self.s1.value())


# ======================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("jebru — NMR analysis (JEOL / Bruker)")
        self.resize(1400, 860)
        self.settings = QtCore.QSettings("jebru", "jebru")
        self.spec: io.Spectrum1D | None = None
        self.pr: P.Processed | None = None
        self.prm = P.ProcParams()
        self.regions: list[A.Region] = []
        self.peaks: list[A.Peak] = []
        self.region_items: list[pg.LinearRegionItem] = []
        self.peak_items = []
        self._drag_item = None
        self._ref_region: A.Region | None = None
        self._ref_value: float | None = None
        self.peak_threshold: float | None = None
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._undo_depth = 0
        self._restoring = False
        self._settled_view = None
        self._view_timer = QtCore.QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(350)
        self._view_timer.timeout.connect(self._flush_view)
        self._build_ui()
        self.vb.sigRangeChanged.connect(self._on_range_changed)
        geo, st = self.settings.value("geometry"), self.settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if st is not None:
            self.restoreState(st)
        self._update_undo_actions()
        self.setAcceptDrops(True)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        # plot
        self.vb = SpectrumViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.invertX(True)
        self.plot.setLabel("bottom", "δ / ppm")
        self.plot.getAxis("left").setStyle(showValues=False)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.curve = self.plot.plot([], [], pen=pg.mkPen("k", width=1))
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)
        self.int_curves = []
        self.bl_curve = self.plot.plot([], [], pen=pg.mkPen((0, 150, 60), width=1.5))
        self.bl_curve.setZValue(5)
        self.bl_scatter = pg.ScatterPlotItem(symbol="o", size=9, brush=pg.mkBrush(0, 170, 70),
                                             pen=pg.mkPen("w", width=1))
        self.bl_scatter.setZValue(6)
        self.plot.addItem(self.bl_scatter)
        self.bl_region_items = []
        self.lbl_items = []
        self.thr_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen((220, 60, 60), style=Qt.DashLine))
        self.thr_line.setVisible(False)
        self.thr_line.sigPositionChangeFinished.connect(lambda *_: self._thr_moved())
        self.plot.addItem(self.thr_line)
        self.cursor_lbl = pg.TextItem("", anchor=(1, 0), color=(80, 80, 80))
        self.plot.addItem(self.cursor_lbl, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.vb.sigDragRegion.connect(self._drag_region)
        self.vb.sigClick.connect(self._plot_click)
        self.setCentralWidget(self.plot)

        # toolbar
        tb = self.addToolBar("main")
        tb.setObjectName("main_toolbar")
        self.tb = tb
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._act(tb, "Open…", self.open_file, "Ctrl+O")
        self._act(tb, "Open folder…", self.open_dir, "Ctrl+Shift+O")
        tb.addSeparator()
        self.act_undo = self._act(tb, "Undo", self.undo, "Ctrl+Z")
        self.act_redo = self._act(tb, "Redo", self.redo, "Ctrl+Shift+Z")
        self.act_redo.setShortcuts([QtGui.QKeySequence("Ctrl+Shift+Z"), QtGui.QKeySequence("Ctrl+Y")])
        tb.addSeparator()
        self.mode_group = QtGui.QActionGroup(self)
        for m in MODES:
            a = QtGui.QAction(MODE_LABEL[m], self, checkable=True)
            a.setData(m)
            a.triggered.connect(lambda _=False, m=m: self.set_mode(m))
            self.mode_group.addAction(a)
            tb.addAction(a)
            if m == "zoom":
                a.setChecked(True)
        tb.addSeparator()
        self._act(tb, "Full view", self.full_view, "F")
        self._act(tb, "Auto phase", self.auto_phase, "A")
        self._act(tb, "Manual phase…", self.manual_phase, "P")
        self._act(tb, "Auto integrate", self.auto_integrate, "I")
        self._act(tb, "Auto peaks", self.auto_peaks, "K")
        self._act(tb, "Clear integrals", self.clear_regions)
        tb.addSeparator()
        self._act(tb, "Copy report", self.copy_report, "Ctrl+Shift+C")

        # menus
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Open…", self.open_file)
        fm.addAction("Open folder…", self.open_dir)
        self.recent_menu = fm.addMenu("Recent files")
        self._update_recent()
        fm.addSeparator()
        em = fm.addMenu("Export")
        em.addAction("Figure (PNG/PDF/SVG, publication)…", self.export_figure)
        em.addAction("Screen image…", self.export_screen)
        em.addAction("Peak list CSV…", self.export_peaks)
        em.addAction("Integrals & multiplets CSV…", self.export_integrals)
        em.addAction("Spectrum CSV (ppm, intensity)…", self.export_spectrum)
        em.addAction("Report text…", self.export_report)
        fm.addSeparator()
        fm.addAction("Quit", self.close, "Ctrl+Q")
        edm = mb.addMenu("Edit")
        edm.addAction(self.act_undo)
        edm.addAction(self.act_redo)
        hm = mb.addMenu("Help")
        hm.addAction("How to use", self.show_help, "F1")

        # side panel: closable (F9 / View menu), movable, floatable, any width
        dock = QtWidgets.QDockWidget("Analysis", self)
        dock.setObjectName("analysis_dock")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
                         | QtWidgets.QDockWidget.DockWidgetClosable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumSize(200, 120)
        dock.setWidget(self.tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [430], Qt.Horizontal)
        self.dock = dock
        act_panel = dock.toggleViewAction()
        act_panel.setText("Analysis panel")
        act_panel.setShortcut(QtGui.QKeySequence("F9"))
        act_panel.setToolTip("Show / hide the analysis panel (F9)")
        vm = QtWidgets.QMenu("View", self)
        vm.addAction(act_panel)
        vm.addAction("Dock panel on the right (reset layout)", self.reset_layout)
        vm.addSeparator()
        vm.addAction(self.tb.toggleViewAction())
        self.tb.toggleViewAction().setText("Toolbar")
        mb.insertMenu(hm.menuAction(), vm)
        tb.addSeparator()
        tb.addAction(act_panel)
        self._build_proc_tab()
        self._build_int_tab()
        self._build_peak_tab()
        self._build_report_tab()
        self._build_info_tab()

        self.status = self.statusBar()
        self.status.showMessage("Open a file or drag & drop it here (.jdf / Bruker folder / .zip)")

    def _act(self, tb, text, slot, key=None):
        a = QtGui.QAction(text, self)
        a.triggered.connect(slot)
        if key:
            a.setShortcut(QtGui.QKeySequence(key))
            a.setToolTip(f"{text} ({key})")
        tb.addAction(a)
        return a

    def _build_proc_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.sp_lb = QtWidgets.QDoubleSpinBox()
        self.sp_lb.setRange(-1, 100)
        self.sp_lb.setDecimals(2)
        self.sp_lb.setSingleStep(0.1)
        self.sp_lb.setSpecialValueText("auto")
        self.sp_lb.setValue(-1)
        f.addRow("Line broadening LB (Hz)", self.sp_lb)
        self.cb_zf = QtWidgets.QComboBox()
        for z in (1, 2, 4, 8):
            self.cb_zf.addItem(f"×{z}", z)
        self.cb_zf.setCurrentIndex(1)
        f.addRow("Zero filling", self.cb_zf)
        self.sp_p0 = QtWidgets.QDoubleSpinBox()
        self.sp_p0.setRange(-720, 720)
        self.sp_p0.setDecimals(1)
        self.sp_p1 = QtWidgets.QDoubleSpinBox()
        self.sp_p1.setRange(-5000, 5000)
        self.sp_p1.setDecimals(1)
        f.addRow("Phase p0 (°)", self.sp_p0)
        f.addRow("Phase p1 (°)", self.sp_p1)
        self.lbl_pivot = QtWidgets.QLabel("-")
        f.addRow("Pivot (ppm)", self.lbl_pivot)
        self.ck_bl = QtWidgets.QCheckBox("Baseline correction")
        self.ck_bl.setChecked(True)
        self.cb_blm = QtWidgets.QComboBox()
        self.cb_blm.addItem("Spline", "spline")
        self.cb_blm.addItem("Polynomial", "poly")
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.ck_bl)
        h.addWidget(self.cb_blm)
        f.addRow(h)
        self.cb_solv = QtWidgets.QComboBox()
        self.cb_solv.addItem("(from file)", "")
        for s in P.SOLVENTS:
            self.cb_solv.addItem(s, s)
        f.addRow("Solvent", self.cb_solv)
        self.cb_ref = QtWidgets.QComboBox()
        self.cb_ref.addItem("Auto (solvent signal)", "auto")
        self.cb_ref.addItem("None (instrument axis)", "none")
        self.cb_ref.addItem("Manual (Reference mode)", "manual")
        f.addRow("Chemical shift reference", self.cb_ref)
        self.lbl_ref = QtWidgets.QLabel("-")
        self.lbl_ref.setWordWrap(True)
        f.addRow("", self.lbl_ref)
        b = QtWidgets.QPushButton("Reprocess")
        b.clicked.connect(lambda: self.reprocess(keep_phase=True))
        b2 = QtWidgets.QPushButton("Reprocess with auto phase")
        b2.clicked.connect(self.auto_phase)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(b)
        h2.addWidget(b2)
        f.addRow(h2)
        gb = QtWidgets.QGroupBox("Manual baseline (Baseline mode)")
        g = QtWidgets.QGridLayout(gb)
        self.cb_blman = QtWidgets.QComboBox()
        self.cb_blman.addItem("Spline (PCHIP)", "spline")
        self.cb_blman.addItem("Polynomial", "poly")
        self.sp_blord = QtWidgets.QSpinBox()
        self.sp_blord.setRange(0, 12)
        self.sp_blord.setValue(3)
        self.sp_blord.setPrefix("order ")
        self.cb_blman.currentIndexChanged.connect(lambda *_: self._set_bl_method())
        self.sp_blord.valueChanged.connect(lambda *_: self._set_bl_method())
        g.addWidget(QtWidgets.QLabel("Fit"), 0, 0)
        g.addWidget(self.cb_blman, 0, 1)
        g.addWidget(self.sp_blord, 0, 2)
        bauto = QtWidgets.QPushButton("Auto points")
        bauto.clicked.connect(lambda: self.auto_bl_points())
        bclr = QtWidgets.QPushButton("Clear (back to auto)")
        bclr.clicked.connect(lambda: self.clear_bl_points())
        bmode = QtWidgets.QPushButton("Baseline mode")
        bmode.clicked.connect(lambda: self._activate_mode("baseline"))
        g.addWidget(bmode, 1, 0)
        g.addWidget(bauto, 1, 1)
        g.addWidget(bclr, 1, 2)
        self.lbl_bl = QtWidgets.QLabel("no manual points (automatic baseline)")
        self.lbl_bl.setStyleSheet("color:#666")
        g.addWidget(self.lbl_bl, 2, 0, 1, 3)
        f.addRow(gb)
        self.ck_implbl = QtWidgets.QCheckBox("Label solvent && impurity signals on the spectrum")
        self.ck_implbl.setChecked(True)
        self.ck_implbl.toggled.connect(lambda *_: self._draw_labels())
        f.addRow(self.ck_implbl)
        self.sp_snr = QtWidgets.QDoubleSpinBox()
        self.sp_snr.setRange(2, 1000)
        self.sp_snr.setValue(6)
        f.addRow("Detection S/N threshold", self.sp_snr)
        self.lbl_log = QtWidgets.QLabel("")
        self.lbl_log.setWordWrap(True)
        self.lbl_log.setStyleSheet("color:#666")
        f.addRow(self.lbl_log)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        sc.setWidget(w)
        self.tabs.addTab(sc, "Processing")

    def _build_int_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.tbl_int = QtWidgets.QTableWidget(0, 7)
        self.tbl_int.setHorizontalHeaderLabels(["Use", "δ (ppm)", "Range", "nH", "Mult.", "J (Hz)", "Note"])
        self.tbl_int.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_int.horizontalHeader().setStretchLastSection(True)
        self.tbl_int.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_int.itemChanged.connect(self._int_item_changed)
        self.tbl_int.itemSelectionChanged.connect(self._int_selected)
        v.addWidget(self.tbl_int)
        h = QtWidgets.QHBoxLayout()
        for text, slot in (("Delete selected", self.delete_selected_region), ("Auto integrate", self.auto_integrate),
                           ("Re-analyse multiplets", self.reanalyze), ("Clear all", self.clear_regions)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            h.addWidget(b)
        v.addLayout(h)
        tip = QtWidgets.QLabel("• Integrate mode: drag to add a region; drag the edges to adjust.\n"
                               "• Editing nH re-normalises all integrals to that value.\n"
                               "• Multiplicity and J can be edited directly (used in the report).")
        tip.setStyleSheet("color:#666")
        v.addWidget(tip)
        self.tabs.addTab(w, "Integrals")

    def _build_peak_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.tbl_pk = QtWidgets.QTableWidget(0, 4)
        self.tbl_pk.setHorizontalHeaderLabels(["δ (ppm)", "Hz", "Intensity", "Note"])
        self.tbl_pk.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_pk.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_pk.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.tbl_pk)
        h = QtWidgets.QHBoxLayout()
        for text, slot in (("Auto peaks", self.auto_peaks), ("Delete selected", self.delete_selected_peaks),
                           ("Clear all", self.clear_peaks)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            h.addWidget(b)
        v.addLayout(h)
        self.ck_peaklbl = QtWidgets.QCheckBox("Show labels on spectrum")
        self.ck_peaklbl.setChecked(False)
        self.ck_peaklbl.toggled.connect(self._draw_peaks)
        v.addWidget(self.ck_peaklbl)
        tip = QtWidgets.QLabel("• Peak mode: click to add / Shift+click to remove the nearest peak.\n"
                               "• Red dashed line = detection threshold (drag it to re-pick peaks).")
        tip.setStyleSheet("color:#666")
        v.addWidget(tip)
        self.tabs.addTab(w, "Peaks")

    def _build_report_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.txt_report = QtWidgets.QPlainTextEdit()
        self.txt_report.setFont(QtGui.QFont("Monospace", 10))
        v.addWidget(self.txt_report)
        self.lbl_imp = QtWidgets.QLabel("")
        self.lbl_imp.setWordWrap(True)
        self.lbl_imp.setStyleSheet("color:#7a2a9a")
        self.lbl_imp.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_imp)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Copy")
        b.clicked.connect(self.copy_report)
        h.addWidget(b)
        b2 = QtWidgets.QPushButton("Regenerate")
        b2.clicked.connect(self.update_report)
        h.addWidget(b2)
        self.sp_dig = QtWidgets.QSpinBox()
        self.sp_dig.setRange(1, 4)
        self.sp_dig.setValue(2)
        self.sp_dig.setPrefix("δ decimals: ")
        self.sp_dig.valueChanged.connect(self.update_report)
        h.addWidget(self.sp_dig)
        v.addLayout(h)
        self.tabs.addTab(w, "Report")

    def _build_info_tab(self):
        self.txt_info = QtWidgets.QPlainTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setFont(QtGui.QFont("Monospace", 9))
        self.tabs.addTab(self.txt_info, "Info")

    def reset_layout(self):
        self.dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.show()
        self.tb.show()
        self.resizeDocks([self.dock], [430], Qt.Horizontal)

    def closeEvent(self, e):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(e)

    # ------------------------------------------------------------ files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.load(urls[0].toLocalFile())

    def open_file(self):
        d = self.settings.value("lastdir", os.path.expanduser("~"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open NMR data", d,
            "NMR data (*.jdf *.zip fid acqus);;JEOL Delta (*.jdf);;Bruker zip (*.zip);;All files (*)")
        if path:
            self.load(path)

    def open_dir(self):
        d = self.settings.value("lastdir", os.path.expanduser("~"))
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Bruker experiment folder", d)
        if path:
            self.load(path)

    def _choose_experiment(self, path):
        """If a folder holds several Bruker experiments (or JEOL files), let the user pick one."""
        if not os.path.isdir(path) or os.path.isfile(os.path.join(path, "acqus")):
            return path
        exps = io.list_bruker_experiments(path)
        jdfs = sorted(os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(".jdf"))
        items = []
        for e in exps:
            info = ""
            try:
                import nmrglue as ng
                a = ng.bruker.read_acqus_file(e)["acqus"]
                info = f"  [{a.get('NUC1', '')} {a.get('PULPROG', '')}]"
            except Exception:
                pass
            if not os.path.isfile(os.path.join(e, "fid")):
                info += " (2D)"
            items.append((os.path.relpath(e, path) + info, e))
        items += [(os.path.relpath(j, path), j) for j in jdfs]
        if len(items) <= 1:
            return items[0][1] if items else path
        labels = [i[0] for i in items]
        sel, ok = QtWidgets.QInputDialog.getItem(self, "Select data", "This folder contains several datasets:", labels, 0, False)
        if not ok:
            return None
        return items[labels.index(sel)][1]

    def load(self, path):
        try:
            path = self._choose_experiment(path)
            if not path:
                return
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            self.spec = io.load(path)
            self.settings.setValue("lastdir", os.path.dirname(path.rstrip("/")))
            self._add_recent(path)
            self.prm = P.ProcParams(lb=self.sp_lb.value(), zf=self.cb_zf.currentData(),
                                    baseline=self.ck_bl.isChecked(), baseline_method=self.cb_blm.currentData(),
                                    ref_mode=self.cb_ref.currentData() if self.cb_ref.currentData() != "manual" else "auto",
                                    solvent=self.cb_solv.currentData(),
                                    bl_manual_method=self.cb_blman.currentData(),
                                    bl_manual_order=self.sp_blord.value())
            self.cb_ref.setCurrentIndex(self.cb_ref.findData(self.prm.ref_mode))
            self.regions, self.peaks = [], []
            self._ref_region, self._ref_value = None, None
            self.peak_threshold = None
            self._run_process()
            self._after_regions()
            self.setWindowTitle(f"jebru — {self.spec.name}  [{self.spec.vendor} {self.spec.nucleus}]")
            self._fill_info()
            self.ck_peaklbl.setChecked(self.spec.nucleus != "1H")
            if self.spec.nucleus == "1H":
                self.auto_integrate()
            self.auto_peaks()
            self.full_view()
            self._undo.clear()
            self._redo.clear()
            self._settled_view = self._view()
            self._update_undo_actions()
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Load error", f"{path}\n\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _add_recent(self, path):
        r = [p for p in (self.settings.value("recent", []) or []) if p != path]
        r.insert(0, path)
        self.settings.setValue("recent", r[:10])
        self._update_recent()

    def _update_recent(self):
        self.recent_menu.clear()
        for p in self.settings.value("recent", []) or []:
            self.recent_menu.addAction(p, lambda p=p: self.load(p))

    def _fill_info(self):
        s = self.spec
        lines = [f"File: {s.path}", f"Vendor: {s.vendor}", f"Nucleus: {s.nucleus}",
                 f"Observe frequency: {s.obs:.4f} MHz", f"Spectral width: {s.sw:.1f} Hz ({s.sw / s.obs:.2f} ppm)",
                 f"Centre: {s.center_ppm:.3f} ppm", f"Solvent: {s.solvent}", f"Data points: {np.asarray(s.data).size}",
                 f"Digital filter group delay: {s.group_delay:.3f} pts", f"Title: {s.title}", "", "--- Parameters ---"]
        for k in sorted(s.params):
            v = str(s.params[k])
            if len(v) < 120:
                lines.append(f"{k} = {v}")
        self.txt_info.setPlainText("\n".join(lines))

    # ------------------------------------------------------------ processing
    def _collect_params(self, keep_phase=True):
        self.prm.lb = self.sp_lb.value()
        self.prm.zf = self.cb_zf.currentData()
        self.prm.baseline = self.ck_bl.isChecked()
        self.prm.baseline_method = self.cb_blm.currentData()
        self.prm.solvent = self.cb_solv.currentData()
        self.prm.ref_mode = self.cb_ref.currentData()
        if keep_phase:
            self.prm.auto_phase = False
            self.prm.p0 = self.sp_p0.value()
            self.prm.p1 = self.sp_p1.value()

    def _run_process(self):
        old_shift = self._axis_shift()
        self.pr = P.process(self.spec, self.prm)
        if self.prm.ref_mode == "auto":
            # store the applied auto shift so that "manual" starts from it
            self.prm.ref_shift = self._axis_shift()
        self.sp_p0.blockSignals(True)
        self.sp_p1.blockSignals(True)
        self.sp_p0.setValue(self.prm.p0)
        self.sp_p1.setValue(self.prm.p1)
        self.sp_p0.blockSignals(False)
        self.sp_p1.blockSignals(False)
        self.lbl_pivot.setText(f"{self.prm.pivot_ppm:.3f}" if self.prm.pivot_ppm is not None else "-")
        self.lbl_ref.setText(self.pr.ref_note or "")
        self.lbl_log.setText(" / ".join(self.pr.log))
        # keep regions/peaks attached to signals when the axis moves
        d = self._axis_shift() - old_shift if old_shift is not None else 0.0
        if d:
            for r in self.regions:
                r.start += d
                r.stop += d
        self._draw_spectrum()

    def _axis_shift(self):
        if self.pr is None:
            return None
        return float((self.pr.ppm[0] + self.pr.ppm[-1]) / 2 - self.spec.center_ppm)

    @undoable
    def reprocess(self, keep_phase=True):
        if not self.spec:
            return
        self._collect_params(keep_phase)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._run_process()
            self._recompute_regions()
            if self.peaks:
                self.auto_peaks(keep_threshold=True)
            self.update_report()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    @undoable
    def auto_phase(self):
        if not self.spec:
            return
        self._collect_params(keep_phase=False)
        self.prm.auto_phase = True
        self.prm.pivot_ppm = None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._run_process()
            self._recompute_regions()
            if self.peaks:
                self.auto_peaks(keep_threshold=True)
            self.update_report()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def manual_phase(self):
        if not self.pr:
            return
        base = self.pr.cplx.copy()
        bp0, bp1 = self.prm.p0, self.prm.p1
        piv = float((self.pr.ppm[0] - self.prm.pivot_ppm) / (self.pr.ppm[0] - self.pr.ppm[-1]))
        dlg = PhaseDialog(bp0, bp1, self)

        def preview(p0, p1):
            y = np.real(P.phase(base, p0 - bp0, p1 - bp1, piv))
            self.curve.setData(self.pr.ppm[::-1], y[::-1])

        dlg.changed.connect(preview)
        self._pivot_line = pg.InfiniteLine(self.prm.pivot_ppm, pen=pg.mkPen((0, 160, 0), width=1))
        self.plot.addItem(self._pivot_line)
        ok = dlg.exec()
        self.plot.removeItem(self._pivot_line)
        if ok:
            self.sp_p0.setValue(dlg.s0.value())
            self.sp_p1.setValue(dlg.s1.value())
            self.reprocess(keep_phase=True)
        else:
            self._draw_spectrum()

    # ------------------------------------------------------------ drawing
    def _draw_spectrum(self):
        if not self.pr:
            return
        # pyqtgraph's clipToView needs ascending x
        if self.vb.mode == "baseline" and self.pr.raw_real is not None:
            self.curve.setData(self.pr.ppm[::-1], self.pr.raw_real[::-1])
        else:
            self.curve.setData(self.pr.ppm[::-1], self.pr.real[::-1])
        self._draw_baseline()
        self._draw_regions()
        self._draw_peaks()

    def full_view(self):
        if not self.pr:
            return
        x, y = self.pr.ppm, self.pr.real
        nuc = self.spec.nucleus
        sig = A.pick_peaks(self.pr, snr=20)
        hi = max((p.ppm for p in sig), default=x[0])
        lo = min((p.ppm for p in sig), default=x[-1])
        if nuc == "1H":
            xr = (min(-0.5, lo - 0.3), max(10.0, hi + 0.5))
        elif nuc == "13C":
            xr = (min(-5.0, lo - 5), max(200.0, hi + 5))
        else:
            pad = (hi - lo) * 0.1 + 5
            xr = (lo - pad, hi + pad)
        xr = (max(xr[0], float(x[-1])), min(xr[1], float(x[0])))
        m = (x >= xr[0]) & (x <= xr[1])
        top = float(y[m].max()) if m.any() else float(y.max())
        if nuc == "13C" and self.peaks:
            top = max(p.height for p in self.peaks if xr[0] <= p.ppm <= xr[1]) \
                if any(xr[0] <= p.ppm <= xr[1] for p in self.peaks) else top
        self.vb.setRange(xRange=xr, yRange=(-0.08 * top, 1.15 * top), padding=0)

    def _draw_regions(self):
        for it in self.region_items:
            self.plot.removeItem(it)
        for c in self.int_curves:
            self.plot.removeItem(c)
        self.region_items, self.int_curves = [], []
        if not self.pr:
            return
        x, y = self.pr.ppm, self.pr.real
        top = y.max()
        span = abs(x[1] - x[0])
        for k, r in enumerate(self.regions):
            it = pg.LinearRegionItem(values=(r.stop, r.start), movable=True,
                                     brush=REGION_BRUSH if r.include else REGION_BRUSH_OFF)
            it.setZValue(-10)
            it.sigRegionChangeFinished.connect(lambda item, k=k: self._region_moved(k, item))
            self.plot.addItem(it)
            self.region_items.append(it)
            sel = (x <= r.start) & (x >= r.stop)
            if sel.sum() > 1:
                cum = np.cumsum(y[sel][::-1])[::-1] * span
                tot = self.regions_total_scale()
                h = 0.25 * top * (cum / tot if tot else cum / max(abs(cum[0]), 1e-30))
                c = self.plot.plot(x[sel], h - h[-1] + 0.02 * top, pen=pg.mkPen((210, 40, 40), width=1.2))
                self.int_curves.append(c)
            lbl = pg.TextItem(f"{r.nH:.2f}" if r.nH else "", anchor=(0.5, 0), color=(20, 80, 200))
            lbl.setPos((r.start + r.stop) / 2, -0.01 * top)
            self.plot.addItem(lbl)
            self.int_curves.append(lbl)

    def regions_total_scale(self):
        vals = [abs(r.value) for r in self.regions if r.include]
        return max(vals) if vals else 0

    def _draw_peaks(self):
        for it in self.peak_items:
            self.plot.removeItem(it)
        self.peak_items = []
        if not self.pr or not self.peaks:
            return
        xs = np.array([p.ppm for p in self.peaks])
        ys = np.array([p.height for p in self.peaks])
        sc = pg.ScatterPlotItem(xs, ys, symbol="t", size=7, brush=pg.mkBrush(230, 120, 0), pen=None)
        self.plot.addItem(sc)
        self.peak_items.append(sc)
        if self.ck_peaklbl.isChecked() and len(self.peaks) <= 300:
            dig = 1 if self.spec.nucleus == "13C" else 3
            for p in self.peaks:
                t = pg.TextItem(f"{p.ppm:.{dig}f}", anchor=(0, 0.5), angle=90, color=(120, 60, 0))
                t.setPos(p.ppm, p.height)
                t.setFont(QtGui.QFont("Sans", 7))
                self.plot.addItem(t)
                self.peak_items.append(t)

    # ------------------------------------------------------------ modes / mouse
    def _activate_mode(self, m):
        for a in self.mode_group.actions():
            if a.data() == m:
                a.setChecked(True)
        self.set_mode(m)

    def set_mode(self, m):
        prev = self.vb.mode
        self.vb.mode = m
        if self.pr is not None and "baseline" in (prev, m) and prev != m:
            self._draw_spectrum()
        self.thr_line.setVisible(m == "peak" and self.pr is not None)
        msg = {
            "zoom": "Zoom: left-drag to zoom in / right-click to zoom out / double-click for full view / "
                    "middle-drag to pan / wheel = vertical scale / Ctrl+Z to undo",
            "integrate": "Integrate: left-drag to add an integral region",
            "peak": "Peak: click to add / Shift+click to remove / drag the red dashed line to change the threshold",
            "baseline": "Baseline: click to add a baseline point / drag to add a baseline region / "
                        "Shift+click to remove / green line = baseline that is subtracted",
            "reference": "Reference: click the reference peak, then enter its chemical shift",
            "phase": "Phase pivot: click to set the pivot for first-order phasing",
        }[m]
        self.status.showMessage(msg)
        if m == "peak" and self.pr is not None:
            thr = self.peak_threshold or self.sp_snr.value() * A.spectrum_noise(self.pr)
            self.thr_line.setValue(thr)

    def _mouse_moved(self, pos):
        if not self.pr:
            return
        p = self.vb.mapSceneToView(pos)
        vr = self.vb.viewRange()
        self.cursor_lbl.setPos(vr[0][0], vr[1][1])
        self.cursor_lbl.setText(f"δ {p.x():.4f} ppm   {p.x() * self.spec.obs:.1f} Hz")

    def _drag_region(self, x0, x1, finished):
        if not self.pr:
            return
        hi, lo = max(x0, x1), min(x0, x1)
        if self._drag_item is None:
            self._drag_item = pg.LinearRegionItem(values=(lo, hi), movable=False, brush=(220, 60, 60, 40))
            self.plot.addItem(self._drag_item)
        self._drag_item.setRegion((lo, hi))
        if finished:
            self.plot.removeItem(self._drag_item)
            self._drag_item = None
            if abs(hi - lo) * self.spec.obs < 0.5:
                return
            if self.vb.mode == "baseline":
                self._add_bl_region(hi, lo)
            else:
                self._add_region(hi, lo)

    @undoable
    def _add_region(self, hi, lo):
        # a new region replaces the ones it overlaps
        self.regions = [q for q in self.regions if q.stop > hi or q.start < lo]
        if self._ref_region not in self.regions:
            self._ref_region = None
        r = A.Region(start=hi, stop=lo)
        r.value = A.integrate(self.pr, hi, lo)
        A.analyze_multiplet(self.pr, r)
        self.regions.append(r)
        self.regions.sort(key=lambda r: -r.start)
        self._normalize()
        self._after_regions()

    @undoable
    def _plot_click(self, x, y):
        if not self.pr:
            return
        mods = QtWidgets.QApplication.keyboardModifiers()
        if self.vb.mode == "baseline":
            if mods & Qt.ShiftModifier:
                self._remove_bl_near(x)
            else:
                raw = x - self.pr.ref_applied
                self.prm.bl_points = sorted(self.prm.bl_points + [float(raw)], reverse=True)
                self._apply_manual_baseline()
            return
        if self.vb.mode == "peak":
            if mods & Qt.ShiftModifier:
                if self.peaks:
                    k = int(np.argmin([abs(p.ppm - x) for p in self.peaks]))
                    self.peaks.pop(k)
            else:
                pk = self._nearest_max(x)
                if pk and not any(abs(p.index - pk.index) <= 1 for p in self.peaks):
                    self.peaks.append(pk)
                    self.peaks.sort(key=lambda p: -p.ppm)
            self._draw_peaks()
            self._fill_peak_table()
            self.update_report()
        elif self.vb.mode == "reference":
            pk = self._nearest_max(x)
            if not pk:
                return
            val, ok = QtWidgets.QInputDialog.getDouble(
                self, "Chemical shift reference", f"Shift of this peak ({pk.ppm:.4f} ppm):", round(pk.ppm, 2), -1000, 1000, 4)
            if ok:
                cur_shift = self._axis_shift()
                delta = val - pk.ppm
                self.prm.ref_mode = "manual"
                self.prm.ref_shift = cur_shift + delta
                self.cb_ref.setCurrentIndex(self.cb_ref.findData("manual"))
                self.reprocess(keep_phase=True)
        elif self.vb.mode == "phase":
            self.prm.pivot_ppm = float(x)
            self.lbl_pivot.setText(f"{x:.3f}")
            self.status.showMessage(f"Pivot = {x:.3f} ppm (used for manual phasing)")

    def _nearest_max(self, x):
        """Local maximum of the spectrum near ppm x (within ±0.02 ppm / ±0.3 ppm for 13C)."""
        w = 0.02 if self.spec.nucleus == "1H" else 0.3
        pk = A.pick_peaks(self.pr, threshold=-np.inf, lo=x - w, hi=x + w)
        if not pk:
            return None
        return max(pk, key=lambda p: p.height - abs(p.ppm - x) * 1e-9)

    @undoable
    def _thr_moved(self):
        self.peak_threshold = float(self.thr_line.value())
        self.auto_peaks(keep_threshold=True)

    # ------------------------------------------------------------ solvent / impurity labels
    RESIDUAL = {"CDCl3": "CHCl3", "DMSO-d6": "DMSO-d5", "CD3OD": "CHD2OD", "D2O": "HOD",
                "Acetone-d6": "acetone-d5", "C6D6": "C6D5H", "CD3CN": "CHD2CN", "CD2Cl2": "CHDCl2",
                "THF-d8": "THF-d7", "Toluene-d8": "toluene-d7", "Pyridine-d5": "pyridine-d4"}

    def _solvent(self):
        return P.norm_solvent(self.prm.solvent or (self.spec.solvent if self.spec else ""))

    def _tag_text(self, tag):
        sv = self._solvent()
        if tag == "solvent":
            return f"{self.RESIDUAL.get(sv, 'solvent')} (residual)"
        if tag == "H2O":
            return "HOD" if sv in ("D2O", "CD3OD") else "H2O"
        return tag

    def _impurity_text(self):
        if not self.pr:
            return ""
        groups = {}
        if self.spec.nucleus == "1H":
            for r in self.regions:
                if r.tag:
                    groups.setdefault(self._tag_text(r.tag), []).append(r.center or (r.start + r.stop) / 2)
        else:
            for p in self.peaks:
                if p.note:
                    groups.setdefault(p.note, []).append(p.ppm)
        if not groups:
            return ""
        dig = 2 if self.spec.nucleus == "1H" else 1
        parts = [f"{k}: " + ", ".join(f"{v:.{dig}f}" for v in sorted(vs, reverse=True))
                 for k, vs in groups.items()]
        return ("Solvent / impurity signals (not in the report unless ticked; '?' = tentative):\n  "
                + "\n  ".join(parts))

    def _draw_labels(self):
        for it in self.lbl_items:
            self.plot.removeItem(it)
        self.lbl_items = []
        if not self.pr or not self.ck_implbl.isChecked():
            return
        x, y = self.pr.ppm, self.pr.real
        top = float(np.max(y)) or 1.0
        marks = []
        if self.spec.nucleus == "1H":
            for r in self.regions:
                if not r.tag:
                    continue
                i0 = int(max(0, np.floor(self.pr.ppm_to_index(r.start))))
                i1 = int(min(len(y) - 1, np.ceil(self.pr.ppm_to_index(r.stop))))
                h = float(y[i0:i1 + 1].max()) if i1 > i0 else 0.0
                marks.append((r.center or (r.start + r.stop) / 2, h, self._tag_text(r.tag)))
        else:
            for p in self.peaks:
                if p.note:
                    marks.append((p.ppm, p.height, p.note))
            sv = self._solvent()
            col = {"13C": 1}.get(self.spec.nucleus)
            if sv and col is not None and P.SOLVENTS.get(sv, (None, None))[col] is not None:
                c = P.SOLVENTS[sv][col]
                if x[-1] <= c <= x[0]:
                    i = int(round(self.pr.ppm_to_index(c)))
                    marks.append((c, float(y[max(0, i - 20):i + 21].max()), sv))
        for px, h, text in marks:
            t = pg.TextItem(text, anchor=(0.5, 1.0), color=(122, 42, 154))
            t.setFont(QtGui.QFont("Sans", 8))
            t.setPos(px, min(h, 1.05 * top) + 0.01 * top)
            t.setZValue(20)
            self.plot.addItem(t)
            self.lbl_items.append(t)

    # ------------------------------------------------------------ manual baseline
    def _draw_baseline(self):
        for it in self.bl_region_items:
            self.plot.removeItem(it)
        self.bl_region_items = []
        show = self.pr is not None and self.vb.mode == "baseline"
        self.bl_curve.setVisible(show)
        self.bl_scatter.setVisible(show)
        if self.pr is not None:
            n = len(self.prm.bl_points) + len(self.prm.bl_regions)
            self.lbl_bl.setText(f"{len(self.prm.bl_points)} points, {len(self.prm.bl_regions)} regions"
                                if n else "no manual points (automatic baseline)")
        if not show:
            return
        x = self.pr.ppm
        bl = self.pr.baseline if self.pr.baseline is not None else np.zeros_like(x)
        self.bl_curve.setData(x[::-1], bl[::-1])
        sh = self.pr.ref_applied
        xs, ys = P.baseline_anchors(x - sh, self.pr.raw_real, self.prm.bl_points, [], self.pr.hz_per_pt)
        self.bl_scatter.setData(xs + sh, ys)
        for hi, lo in self.prm.bl_regions:
            it = pg.LinearRegionItem(values=(lo + sh, hi + sh), movable=False,
                                     brush=pg.mkBrush(0, 170, 70, 40), pen=pg.mkPen((0, 150, 60)))
            it.setZValue(-9)
            self.plot.addItem(it)
            self.bl_region_items.append(it)

    def _apply_manual_baseline(self):
        """Fast path: recompute only the baseline from the stored raw spectrum."""
        if self.pr is None:
            return
        if not (self.prm.bl_points or self.prm.bl_regions):
            # back to the automatic baseline: full reprocess (phase kept)
            self.prm.auto_phase = False
            self._run_process()
        else:
            raw_ppm = self.pr.ppm - self.pr.ref_applied
            bl = P.manual_baseline(raw_ppm, self.pr.raw_real, self.prm.bl_points, self.prm.bl_regions,
                                   self.prm.bl_manual_method, self.prm.bl_manual_order, self.pr.hz_per_pt)
            self.pr.baseline = bl
            self.pr.real = self.pr.raw_real - bl
            self.lbl_log.setText(" / ".join(l for l in self.pr.log if not l.startswith(("baseline", "manual")))
                                 + f" / manual baseline ({len(self.prm.bl_points)} pts, "
                                 f"{len(self.prm.bl_regions)} regions)")
            self._draw_spectrum()
        for p in self.peaks:
            if 0 <= p.index < len(self.pr.real):
                p.height = float(self.pr.real[p.index])
        self._recompute_regions()
        self._draw_peaks()
        self.update_report()

    @undoable
    def _add_bl_region(self, hi, lo):
        sh = self.pr.ref_applied
        self.prm.bl_regions = self.prm.bl_regions + [(float(hi - sh), float(lo - sh))]
        self._apply_manual_baseline()

    def _remove_bl_near(self, x):
        raw = x - self.pr.ref_applied
        for k, (hi, lo) in enumerate(self.prm.bl_regions):
            if lo <= raw <= hi:
                self.prm.bl_regions = self.prm.bl_regions[:k] + self.prm.bl_regions[k + 1:]
                self._apply_manual_baseline()
                return
        if self.prm.bl_points:
            k = int(np.argmin([abs(p - raw) for p in self.prm.bl_points]))
            self.prm.bl_points = self.prm.bl_points[:k] + self.prm.bl_points[k + 1:]
            self._apply_manual_baseline()

    @undoable
    def auto_bl_points(self):
        if self.pr is None:
            return
        raw_ppm = self.pr.ppm - self.pr.ref_applied
        self.prm.bl_points = P.auto_baseline_points(raw_ppm, self.pr.raw_real, self.pr.hz_per_pt)
        self.prm.bl_regions = []
        self._activate_mode("baseline")
        self._apply_manual_baseline()

    @undoable
    def clear_bl_points(self):
        if self.pr is None:
            return
        self.prm.bl_points, self.prm.bl_regions = [], []
        self._apply_manual_baseline()

    @undoable
    def _set_bl_method(self):
        if self.pr is None:
            return
        self.prm.bl_manual_method = self.cb_blman.currentData()
        self.prm.bl_manual_order = self.sp_blord.value()
        self.sp_blord.setEnabled(self.prm.bl_manual_method == "poly")
        if self.prm.bl_points or self.prm.bl_regions:
            self._apply_manual_baseline()

    # ------------------------------------------------------------ integrals
    @undoable
    def auto_integrate(self):
        if not self.pr:
            return
        snr = self.sp_snr.value()
        if self.spec.nucleus == "1H":
            regs, _ = A.build_regions(self.pr, snr=snr)
        else:
            regs = A.auto_regions(self.pr, snr=snr)
            for r in regs:
                A.analyze_multiplet(self.pr, r)
        self.regions = regs
        self._ref_region, self._ref_value = None, None
        self._normalize()
        self._after_regions()

    @undoable
    def reanalyze(self):
        if not self.pr:
            return
        for r in self.regions:
            r.value = A.integrate(self.pr, r.start, r.stop)
            A.analyze_multiplet(self.pr, r)
        self._normalize()
        self._after_regions()

    def _recompute_regions(self):
        if not self.regions:
            return
        for r in self.regions:
            r.value = A.integrate(self.pr, r.start, r.stop)
            A.analyze_multiplet(self.pr, r)
        self._normalize()
        self._after_regions()

    def _normalize(self):
        if self._ref_region is not None and self._ref_region in self.regions:
            A.normalize(self.regions, self._ref_region, self._ref_value)
        else:
            A.normalize(self.regions)

    def _after_regions(self):
        self._draw_regions()
        self._fill_int_table()
        self.update_report()

    @undoable
    def _region_moved(self, k, item):
        if k >= len(self.regions):
            return
        lo, hi = item.getRegion()
        r = self.regions[k]
        r.start, r.stop = hi, lo
        r.value = A.integrate(self.pr, hi, lo)
        A.analyze_multiplet(self.pr, r)
        self._normalize()
        QtCore.QTimer.singleShot(0, self._after_regions)

    def _fill_int_table(self):
        t = self.tbl_int
        t.blockSignals(True)
        t.setRowCount(len(self.regions))
        for i, r in enumerate(self.regions):
            ck = QtWidgets.QTableWidgetItem()
            ck.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            ck.setCheckState(Qt.Checked if r.include else Qt.Unchecked)
            t.setItem(i, 0, ck)
            cen = r.center if r.center else (r.start + r.stop) / 2
            self._ro(t, i, 1, f"{cen:.3f}")
            self._ro(t, i, 2, f"{r.start:.3f}–{r.stop:.3f}")
            t.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{r.nH:.2f}"))
            t.setItem(i, 4, QtWidgets.QTableWidgetItem(r.mult))
            t.setItem(i, 5, QtWidgets.QTableWidgetItem(", ".join(f"{j:.1f}" for j in r.J)))
            self._ro(t, i, 6, self._tag_text(r.tag))
        t.blockSignals(False)

    @staticmethod
    def _ro(t, i, j, text):
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        t.setItem(i, j, it)

    @undoable
    def _int_item_changed(self, item):
        i, j = item.row(), item.column()
        if i >= len(self.regions):
            return
        r = self.regions[i]
        if j == 0:
            r.include = item.checkState() == Qt.Checked
            self._normalize()
            self._after_regions()
        elif j == 3:
            try:
                v = float(item.text())
            except ValueError:
                return
            if v > 0:
                self._ref_region, self._ref_value = r, v
                self._normalize()
                self._after_regions()
        elif j == 4:
            r.mult = item.text().strip() or "m"
            if r.mult == "m":
                r.J = []
            self.update_report()
        elif j == 5:
            try:
                r.J = [float(v) for v in item.text().replace(",", " ").split()]
            except ValueError:
                pass
            self.update_report()

    def _int_selected(self):
        rows = {i.row() for i in self.tbl_int.selectedIndexes()}
        for k, it in enumerate(self.region_items):
            r = self.regions[k]
            base = REGION_BRUSH if r.include else REGION_BRUSH_OFF
            it.setBrush(pg.mkBrush(250, 180, 30, 70) if k in rows else pg.mkBrush(*base))

    @undoable
    def delete_selected_region(self):
        rows = sorted({i.row() for i in self.tbl_int.selectedIndexes()}, reverse=True)
        for r in rows:
            if self.regions[r] is self._ref_region:
                self._ref_region = None
            self.regions.pop(r)
        self._normalize()
        self._after_regions()

    @undoable
    def clear_regions(self):
        self.regions = []
        self._ref_region = None
        self._after_regions()

    # ------------------------------------------------------------ peaks
    @undoable
    def auto_peaks(self, keep_threshold=False):
        if not self.pr:
            return
        thr = self.peak_threshold if keep_threshold else None
        if self.spec.nucleus == "1H":
            if self.regions and not keep_threshold:
                # lines inside the integral regions only
                self.peaks = [p for r in self.regions for p in
                              A.pick_peaks(self.pr, lo=r.stop, hi=r.start, snr=self.sp_snr.value())
                              if p.height > 0.04 * max(q.height for q in r.lines or [p])]
            else:
                self.peaks = A.pick_peaks(self.pr, threshold=thr, snr=self.sp_snr.value())
        else:
            self.peaks = A.carbon_peaks(self.pr, snr=self.sp_snr.value(), threshold=thr)
            A.tag_impurities_13c(self.pr, self.peaks)
        self.peaks.sort(key=lambda p: -p.ppm)
        if self.vb.mode == "peak":
            self.thr_line.setValue(thr if thr is not None else self.sp_snr.value() * A.spectrum_noise(self.pr))
        self._draw_peaks()
        self._fill_peak_table()
        self.update_report()

    def _fill_peak_table(self):
        t = self.tbl_pk
        t.setRowCount(len(self.peaks))
        dig = 2 if self.spec.nucleus == "13C" else 4
        for i, p in enumerate(self.peaks):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{p.ppm:.{dig}f}"))
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{p.ppm * self.spec.obs:.2f}"))
            t.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{p.height:.4g}"))
            t.setItem(i, 3, QtWidgets.QTableWidgetItem(p.note))

    @undoable
    def delete_selected_peaks(self):
        rows = sorted({i.row() for i in self.tbl_pk.selectedIndexes()}, reverse=True)
        for r in rows:
            self.peaks.pop(r)
        self._draw_peaks()
        self._fill_peak_table()
        self.update_report()

    @undoable
    def clear_peaks(self):
        self.peaks = []
        self._draw_peaks()
        self._fill_peak_table()
        self.update_report()

    # ------------------------------------------------------------ undo / redo
    def _view(self):
        (x0, x1), (y0, y1) = self.vb.viewRange()
        return (float(x0), float(x1), float(y0), float(y1))

    @staticmethod
    def _view_differs(a, b):
        if a is None or b is None:
            return False
        sx = abs(a[1] - a[0]) or 1.0
        sy = abs(a[3] - a[2]) or 1.0
        return (abs(a[0] - b[0]) > 1e-6 * sx or abs(a[1] - b[1]) > 1e-6 * sx or
                abs(a[2] - b[2]) > 1e-6 * sy or abs(a[3] - b[3]) > 1e-6 * sy)

    def _snapshot(self, view=None):
        ref_idx = next((i for i, r in enumerate(self.regions) if r is self._ref_region), None)
        return dict(view=view or self._view(), prm=copy.deepcopy(self.prm),
                    regions=copy.deepcopy(self.regions), ref_idx=ref_idx, ref_value=self._ref_value,
                    peaks=copy.deepcopy(self.peaks), thr=self.peak_threshold)

    @staticmethod
    def _same_state(a, b):
        """Same data state (view is ignored)."""
        if a is None or b is None:
            return False

        def rkey(r):
            return (round(r.start, 9), round(r.stop, 9), round(r.nH, 6), r.mult, tuple(r.J), r.include)
        return (a["prm"].to_dict() == b["prm"].to_dict()
                and [rkey(r) for r in a["regions"]] == [rkey(r) for r in b["regions"]]
                and a["ref_idx"] == b["ref_idx"] and a["ref_value"] == b["ref_value"]
                and [(p.index, round(p.ppm, 9)) for p in a["peaks"]] == [(p.index, round(p.ppm, 9)) for p in b["peaks"]]
                and a["thr"] == b["thr"])

    def _push_undo(self, snap):
        self._undo.append(snap)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self._update_undo_actions()

    def _on_range_changed(self, *_):
        if self._restoring or self._undo_depth or self.pr is None:
            return
        self._view_timer.start()

    def _flush_view(self):
        """Record a finished zoom/pan as one undo step."""
        self._view_timer.stop()
        if self.pr is None or self._restoring:
            return
        cur = self._view()
        if self._settled_view is None:
            self._settled_view = cur
            return
        if self._view_differs(self._settled_view, cur):
            self._push_undo(self._snapshot(view=self._settled_view))
            self._settled_view = cur

    def _sync_widgets(self):
        p = self.prm
        for w in (self.sp_lb, self.cb_zf, self.ck_bl, self.cb_blm, self.cb_solv, self.cb_ref):
            w.blockSignals(True)
        self.sp_lb.setValue(p.lb)
        self.cb_zf.setCurrentIndex(max(0, self.cb_zf.findData(p.zf)))
        self.ck_bl.setChecked(p.baseline)
        self.cb_blm.setCurrentIndex(max(0, self.cb_blm.findData(p.baseline_method)))
        self.cb_solv.setCurrentIndex(max(0, self.cb_solv.findData(p.solvent)))
        self.cb_ref.setCurrentIndex(max(0, self.cb_ref.findData(p.ref_mode)))
        for w in (self.sp_lb, self.cb_zf, self.ck_bl, self.cb_blm, self.cb_solv, self.cb_ref):
            w.blockSignals(False)
        for w in (self.cb_blman, self.sp_blord):
            w.blockSignals(True)
        self.cb_blman.setCurrentIndex(max(0, self.cb_blman.findData(p.bl_manual_method)))
        self.sp_blord.setValue(p.bl_manual_order)
        self.sp_blord.setEnabled(p.bl_manual_method == "poly")
        for w in (self.cb_blman, self.sp_blord):
            w.blockSignals(False)

    def _restore(self, snap):
        self._restoring = True
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if snap["prm"].to_dict() != self.prm.to_dict():
                self.prm = copy.deepcopy(snap["prm"])
                self.prm.auto_phase = False
                self._sync_widgets()
                self._run_process()
            self.regions = copy.deepcopy(snap["regions"])
            i = snap["ref_idx"]
            self._ref_region = self.regions[i] if i is not None and i < len(self.regions) else None
            self._ref_value = snap["ref_value"]
            self.peaks = copy.deepcopy(snap["peaks"])
            self.peak_threshold = snap["thr"]
            self._after_regions()
            self._draw_peaks()
            self._fill_peak_table()
            self.update_report()
            x0, x1, y0, y1 = snap["view"]
            self.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
            self._settled_view = self._view()
        finally:
            self._view_timer.stop()
            self._restoring = False
            QtWidgets.QApplication.restoreOverrideCursor()
        self._update_undo_actions()

    def undo(self):
        self._flush_view()
        if not self._undo:
            self.status.showMessage("Nothing to undo", 2000)
            return
        snap = self._undo.pop()
        self._redo.append(self._snapshot())
        self._restore(snap)
        self.status.showMessage(f"Undo ({len(self._undo)} more)", 2000)

    def redo(self):
        self._flush_view()
        if not self._redo:
            self.status.showMessage("Nothing to redo", 2000)
            return
        snap = self._redo.pop()
        self._undo.append(self._snapshot())
        self._restore(snap)
        self.status.showMessage(f"Redo ({len(self._redo)} more)", 2000)

    def _update_undo_actions(self):
        if hasattr(self, "act_undo"):
            self.act_undo.setEnabled(bool(self._undo))
            self.act_redo.setEnabled(bool(self._redo))

    # ------------------------------------------------------------ report / export
    def report_text(self):
        if not self.pr:
            return ""
        dig = self.sp_dig.value()
        if self.spec.nucleus == "1H":
            return A.report_1h(self.pr, self.regions, digits=dig) if self.regions else ""
        if self.spec.nucleus == "13C":
            return A.report_13c(self.pr, self.peaks, digits=max(1, dig - 1))
        vals = ", ".join(f"{p.ppm:.{dig}f}" for p in self.peaks)
        return f"{self.spec.nucleus} NMR ({A.nominal_freq(self.pr)} MHz, {A._solvent_label(self.pr)}) δ {vals}."

    def update_report(self):
        self.txt_report.setPlainText(self.report_text())
        self.lbl_imp.setText(self._impurity_text())
        self._draw_labels()

    def copy_report(self):
        QtWidgets.QApplication.clipboard().setText(self.txt_report.toPlainText())
        self.status.showMessage("Report copied to clipboard", 3000)

    def _save_name(self, title, filt, suffix):
        if not self.spec:
            return None
        d = self.settings.value("lastdir", os.path.expanduser("~"))
        base = os.path.splitext(self.spec.name)[0] or "spectrum"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(d, base + suffix), filt)
        return path or None

    def export_figure(self):
        path = self._save_name("Save figure", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)", ".png")
        if not path:
            return
        from .plotting import plot_spectrum
        xr = self.vb.viewRange()[0]
        plot_spectrum(self.pr, self.regions, self.peaks, path=path, xlim=(max(xr), min(xr)))
        self.status.showMessage(f"Saved: {path}", 5000)

    def export_screen(self):
        path = self._save_name("Save image", "PNG (*.png);;SVG (*.svg)", ".png")
        if not path:
            return
        from pyqtgraph import exporters
        ex = exporters.SVGExporter(self.plot.plotItem) if path.endswith(".svg") else \
            exporters.ImageExporter(self.plot.plotItem)
        if not path.endswith(".svg"):
            ex.parameters()["width"] = 2400
        ex.export(path)

    def export_peaks(self):
        path = self._save_name("Peak list", "CSV (*.csv)", "_peaks.csv")
        if not path:
            return
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ppm", "Hz", "height"])
            for p in self.peaks:
                w.writerow([f"{p.ppm:.4f}", f"{p.ppm * self.spec.obs:.2f}", f"{p.height:.6g}"])

    def export_integrals(self):
        path = self._save_name("Integrals & multiplets", "CSV (*.csv)", "_integrals.csv")
        if not path:
            return
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["from_ppm", "to_ppm", "center_ppm", "nH", "mult", "J_Hz", "raw", "tag", "include"])
            for r in self.regions:
                w.writerow([f"{r.start:.4f}", f"{r.stop:.4f}", f"{r.center:.4f}", f"{r.nH:.3f}", r.mult,
                            " ".join(f"{j:.1f}" for j in r.J), f"{r.value:.6g}", r.tag, int(r.include)])

    def export_spectrum(self):
        path = self._save_name("Spectrum data", "CSV (*.csv)", "_spectrum.csv")
        if not path:
            return
        np.savetxt(path, np.c_[self.pr.ppm, self.pr.real], delimiter=",", header="ppm,intensity",
                   comments="", fmt="%.6f,%.6g")

    def export_report(self):
        path = self._save_name("Report", "Text (*.txt)", "_report.txt")
        if not path:
            return
        with open(path, "w") as fh:
            fh.write(self.txt_report.toPlainText() + "\n")

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "How to use", HELP_TEXT)


HELP_TEXT = """\
[Open] .jdf (JEOL Delta), Bruker experiment folder (fid, acqus) or a .zip of either. Drag & drop works too.
On loading, FT, phasing, baseline correction, solvent referencing, integration and peak picking run automatically.

[Mouse - Zoom mode]
  Left-drag: zoom in    Right-click: zoom out    Double-click: full view
  Middle-drag: pan      Wheel: vertical scale    Ctrl+wheel: horizontal zoom

[Modes]
  Integrate: drag to add a region; drag the region edges to adjust.
  Peak: click to add / Shift+click to remove / red dashed line = threshold.
  Baseline: click = baseline point, drag = baseline region, Shift+click = remove.
             The green line is the baseline that gets subtracted (spline or polynomial,
             set in the Processing tab). "Auto points" suggests points; "Clear" returns
             to the automatic baseline.
  Reference: click a peak and enter its chemical shift (e.g. TMS = 0).
  Phase pivot: set the pivot point for first-order phasing.

[Panel] F9 shows / hides the analysis panel. Drag its title bar to move it (left, right,
        bottom or floating), drag its edge to resize; double-click the title of a floating
        panel to dock it again. View > "Dock panel on the right" resets the layout.

[Undo]  Ctrl+Z undo (zoom, integrals, peaks, phase, referencing, processing)
        Ctrl+Shift+Z / Ctrl+Y redo

[Keys]  A auto phase / P manual phase / I auto integrate / K auto peaks / F full view
"""


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("jebru")
    w = MainWindow()
    w.show()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        QtCore.QTimer.singleShot(0, lambda: w.load(args[0]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
