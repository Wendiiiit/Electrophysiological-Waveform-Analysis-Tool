"""
PCM Waveform Viewer
===================
A desktop application for viewing and comparing PCM waveform data from Excel files.

Dependencies:
    pip install PyQt6 pandas numpy pyqtgraph openpyxl

Usage:
    python pcm_viewer.py
"""

import sys
import numpy as np
import pandas as pd
import math

#Import function from input_output.py
from input_output import build_io_curve_from_table
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QLabel, QFileDialog, QScrollArea, QFrame,
    QCheckBox, QMessageBox, QStatusBar, QToolBar,
    QSizePolicy, QToolTip, QComboBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QAction, QIcon, QCursor

import pyqtgraph as pg


# ── Colour palette ────────────────────────────────────────────────────────────
WAVEFORM_COLORS = [
    "#4e9af1", "#f16c4e", "#4ef18a", "#f1d44e",
    "#c44ef1", "#4ef1e8", "#f14e8a", "#a8f14e",
]

BG_COLOR       = "#F5F3EE"
PANEL_COLOR    = "#ffffff"
BORDER_COLOR   = "#D8D5CF"
TEXT_COLOR      = "#202124"
TABLE_GRID_COLOR = "#A8A8A8"
ACCENT_COLOR   = "#6c85ab"
NEUTRAL_METADATA_COLOR = "#DDE8D5"
VESTIBULAR_COLOR = "#F3D7B8"   
COCHLEAR_COLOR   = "#D7E3F0"  
MERGED_COLOR   = "#ffffff"
VESTIBULAR_PLOT_COLOR = "#B9783E"
COCHLEAR_PLOT_COLOR   = "#5B7FA6"

# ── Data table column colours ────────────────────────────────────────────────
COLUMN_COLORS = {
    "✓": NEUTRAL_METADATA_COLOR,
    "Date/Time": NEUTRAL_METADATA_COLOR,
    "Stim. dur (ms)": NEUTRAL_METADATA_COLOR,
    "Hz (1=mono; 2=bi)": NEUTRAL_METADATA_COLOR,
    "r/f": NEUTRAL_METADATA_COLOR,
    "dB atten.": NEUTRAL_METADATA_COLOR,
    "ISI": NEUTRAL_METADATA_COLOR,
    "cont. dB noise": NEUTRAL_METADATA_COLOR,
    "Desc": NEUTRAL_METADATA_COLOR,
    "Vestibular PtP": VESTIBULAR_COLOR,
    "Cochlear PtP": COCHLEAR_COLOR,
}

# ── Layout constants ──────────────────────────────────────────────────────────
# Columns A–I (0-based: 0–8) are metadata labels
# Columns J–O (9–14) are skipped
# Columns P+ (15 onwards) are PCM data

META_COL_SLICE = slice(0, 9)    # A–I
#Separate Vestibular and Cochlear into two different categories
VESTIBULAR_COL_SLICE = slice(9, 12)  # J-L
COCHLEAR_COL_SLICE = slice(12, 15)   # M-O
PCM_COL_START  = 15             # P (0-based)

# Channel Display Windows 
CHANNEL_WINDOWS = {
    "All Channels": None, 
    "Near-Field VsEP": (0.0, 20.0), 
    "Acceleration": (20.0, 40.0), 
    "Microphone": (40.0, 60.0)
}



#Helper function to handle option values
#For handling missing MIN MAX PtP
def format_optional_value(value) -> str:
    if pd.isna(value):
        return "—"

    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"

def merge_pcm(arrays: list[np.ndarray]) -> np.ndarray:
    """Sum multiple PCM arrays (audio mix), truncating to the shortest length."""
    if not arrays:
        return np.array([])
    min_len = min(len(a) for a in arrays)
    stacked = np.vstack([a[:min_len].astype(float) for a in arrays])
    return stacked.sum(axis=0)

# ── Hover read-out (highlight nearest point + tooltip) ────────────────────────

HOVER_PX_RADIUS = 12   # cursor must be within this many pixels of a point to snap
HOVER_DOT_SIZE  = 9    # diameter (px) of the highlight dot — thicker than the 1px line

def attach_hover_readout(plot: pg.PlotWidget):
    """On mouse-move, highlight the nearest data point (filled dot in the curve's
    colour) and show a persistent in-plot label with its value. Works for single-
    or multi-curve plots."""
    vb = plot.getViewBox()

    highlight = pg.ScatterPlotItem(size=HOVER_DOT_SIZE, pxMode=True)
    highlight.setZValue(1000)          # sit on top of the curves
    highlight.hide()
    plot.addItem(highlight)

    # In-plot label instead of QToolTip — QToolTip auto-hides when the mouse stops.
    readout = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush(255, 255, 255, 210),border=pg.mkPen("#A8A8A8"))
    readout.setZValue(1001)
    readout.hide()
    plot.addItem(readout)

    def on_move(evt):
        pos = evt[0]                   # SignalProxy delivers the args as a tuple
        if not vb.sceneBoundingRect().contains(pos):
            highlight.hide(); readout.hide(); return

        mp = vb.mapSceneToView(pos)
        mx, my = mp.x(), mp.y()
        xps, yps = vb.viewPixelSize()  # data units per pixel, per axis

        best = None                    # (dist_px, x, y, color)
        for item in plot.listDataItems():
            xd, yd = item.getData()
            if xd is None or len(xd) == 0:
                continue
            i = int(np.searchsorted(xd, mx))          # x is sorted (time axis)
            for j in (i - 1, i, i + 1):               # check nearest neighbours
                if 0 <= j < len(xd):
                    d = math.hypot((xd[j] - mx) / xps, (yd[j] - my) / yps)
                    if best is None or d < best[0]:
                        color = pg.mkPen(item.opts["pen"]).color()
                        best = (d, float(xd[j]), float(yd[j]), color)

        if best is None or best[0] > HOVER_PX_RADIUS:
            highlight.hide(); readout.hide(); return

        _, px, py, color = best
        highlight.setData([px], [py], symbol="o", size=HOVER_DOT_SIZE,
                          brush=pg.mkBrush(color), pen=pg.mkPen(color))
        highlight.show()
        readout.setText(f"dB = {px:.4g} dB\n" f"PtP = {py:.4g}", color="#202124")
        readout.setPos(px, py)
        readout.show()

    # SignalProxy rate-limits the mouse-move flood; keep refs alive on the widget.
    plot._hover_proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=60, slot=on_move)
    plot._hover_highlight = highlight
    plot._hover_readout = readout

def attach_hover_readout_deprecated(plot: pg.PlotWidget):
    """On mouse-move, highlight the nearest data point (filled dot in the curve's
    colour) and show a tooltip with its value. Works for single- or multi-curve plots."""
    vb = plot.getViewBox()

    highlight = pg.ScatterPlotItem(size=HOVER_DOT_SIZE, pxMode=True)
    highlight.setZValue(1000)          # sit on top of the curves
    highlight.hide()
    plot.addItem(highlight)

    def on_move(evt):
        pos = evt[0]                   # SignalProxy delivers the args as a tuple
        if not vb.sceneBoundingRect().contains(pos):
            highlight.hide(); QToolTip.hideText(); return

        mp = vb.mapSceneToView(pos)
        mx, my = mp.x(), mp.y()
        xps, yps = vb.viewPixelSize()  # data units per pixel, per axis

        best = None                    # (dist_px, x, y, color)
        for item in plot.listDataItems():
            xd, yd = item.getData()
            if xd is None or len(xd) == 0:
                continue
            i = int(np.searchsorted(xd, mx))          # x is sorted (time axis)
            for j in (i - 1, i, i + 1):               # check nearest neighbours
                if 0 <= j < len(xd):
                    d = math.hypot((xd[j] - mx) / xps, (yd[j] - my) / yps)
                    if best is None or d < best[0]:
                        color = pg.mkPen(item.opts["pen"]).color()
                        best = (d, float(xd[j]), float(yd[j]), color)

        if best is None or best[0] > HOVER_PX_RADIUS:
            highlight.hide(); QToolTip.hideText(); return

        _, px, py, color = best
        highlight.setData([px], [py], symbol="o", size=HOVER_DOT_SIZE,
                          brush=pg.mkBrush(color), pen=pg.mkPen(color))
        highlight.show()
        QToolTip.showText(QCursor.pos(), f"t = {px:.4g} ms<br>value = {py:.4g}")

    # SignalProxy rate-limits the mouse-move flood; keep refs alive on the widget.
    plot._hover_proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=60, slot=on_move)
    plot._hover_highlight = highlight

class XZoomViewBox(pg.ViewBox):
    """ViewBox that zooms only the horizontal axis on mouse-wheel / two-finger
    scroll, centered on the cursor. The vertical axis is left untouched."""

    def wheelEvent(self, ev, axis=None):
        # Reuse pyqtgraph's own scale factor so direction/feel match its
        # convention: wheel-up (delta > 0) -> s < 1 -> zoom in.
        s = 1.02 ** (ev.delta() * self.state["wheelScaleFactor"])

        # Cursor position in data coords, so the zoom stays centered on it.
        inv, _ = self.childGroup.transform().inverted()
        center = pg.Point(inv.map(ev.pos()))

        self._resetTarget()
        self.scaleBy(x=s, y=1.0, center=center)   # y=1.0 -> vertical axis unchanged
        ev.accept()                               # don't bubble up to the scroll area

# ── Waveform plot widget ──────────────────────────────────────────────────────

class WaveformPlot(pg.PlotWidget):
    def __init__(self, title: str, color: str, pcm: np.ndarray, time_ms: np.ndarray, parent=None):
        super().__init__(parent=parent, background=PANEL_COLOR)

        #print(pcm)
        self.setMinimumHeight(110)
        self.setMaximumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Axes styling
        self.getAxis("bottom").setTextPen(pg.mkPen(TEXT_COLOR))
        #self.getAxis("bottom").setTicks([[(v, str(v)) for v in time_ms]]) 
        self.getAxis("bottom").setLabel("ms")      
        self.getAxis("left").setTextPen(pg.mkPen(TEXT_COLOR))
        for ax in ("bottom", "left", "top", "right"):
            self.getAxis(ax).setPen(pg.mkPen(BORDER_COLOR))

        self.setTitle(title, color=color, size="10pt")
        self.showGrid(x=False, y=True, alpha=0.15)

        pen = pg.mkPen(color=color, width=1)

        #print("time points:", len(time_ms))
        #print("PCM samples:", len(pcm))

        self.plot(time_ms, pcm, pen=pen, antialias=True)

        # Horizontal zero line
        self.addLine(y=0, pen=pg.mkPen(color=BORDER_COLOR, style=Qt.PenStyle.DashLine))

        # Disable all mouse interaction — scroll wheel scrolls the list instead
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        attach_hover_readout(self)


# ── Waveform panel (scrollable stack of plots) ───────────────────────────────

class WaveformPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background: {BG_COLOR}; border: none;")

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self.setWidget(self._container)

        self._plots: list[WaveformPlot] = []

    def clear(self):
        for p in self._plots:
            self._layout.removeWidget(p)
            p.deleteLater()
        self._plots.clear()

    def add_waveform(self, title: str, color: str, pcm: np.ndarray, time_ms: np.ndarray):
        plot = WaveformPlot(title, color, pcm, time_ms)
        self._layout.insertWidget(self._layout.count() - 1, plot)
        self._plots.append(plot)

    def rebuild(self, selected_rows: list[dict]):
        self.clear()
        for row in selected_rows:
            self.add_waveform(row["label"], row["color"], row["pcm"], row["time_ms"])
    
    #Merged waveform graph, creates one single plot and draws all selected waveform on top of each other 
    def show_merged(self, selected_rows: list[dict]):
        self.clear()
        plot = pg.PlotWidget(background=PANEL_COLOR, viewBox=XZoomViewBox())

        # plot = pg.PlotWidget(background=PANEL_COLOR)
        plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot.getAxis("bottom").setTextPen(pg.mkPen(TEXT_COLOR))
        plot.getAxis("bottom").setLabel("ms")
        plot.getAxis("left").setTextPen(pg.mkPen(TEXT_COLOR))
        for ax in ("bottom", "left", "top", "right"):
            plot.getAxis(ax).setPen(pg.mkPen(BORDER_COLOR))
        plot.showGrid(x=False, y=True, alpha=0.15)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.addLine(y=0, pen=pg.mkPen(color=BORDER_COLOR, style=Qt.PenStyle.DashLine))

        for row in selected_rows:
            pen = pg.mkPen(color=row["color"], width=1)
            plot.plot(row["time_ms"], row["pcm"], pen=pen, antialias=True)

        attach_hover_readout(plot)
        self._layout.insertWidget(self._layout.count() - 1, plot)
        self._plots.append(plot)

    #Display External Plot 
    def show_plot_widget(self, plot):
        self.clear()

        self._layout.insertWidget(
            self._layout.count() - 1,
            plot
        )

        self._plots.append(plot)

# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCM Waveform Viewer")
        self.resize(1280, 820)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG_COLOR}; color: {TEXT_COLOR}; }}
            QTableWidget {{
                background: {PANEL_COLOR};
                gridline-color: {TABLE_GRID_COLOR};
                border: 1px solid {TABLE_GRID_COLOR};
                color: {TEXT_COLOR};
                font-size: 12px;
            }}
            QTableWidget::indicator {{
                width: 18px;
                height: 18px;
            }}
            QTableWidget::indicator:unchecked {{
                border: 2px solid #7A7A7A;
                background-color: #FFFFFF;
                border-radius: 2px;
            }}
            QTableWidget::indicator:unchecked:hover {{
                border: 2px solid #202124;
                background-color: #F5F3EE;
            }}
            QHeaderView::section {{
                background: {BORDER_COLOR};
                color: {TEXT_COLOR};
                padding: 4px;
                border: none;
                font-weight: bold;
                font-size: 11px;
            }}
            QTableWidget::item:selected {{
                background: #2a3a55;
            }}
            QPushButton {{
                background: {ACCENT_COLOR};
                color: #fff;
                border: none;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: #6aaef3; }}
            QPushButton:pressed {{ background: #3a7acc; }}
            QPushButton#secondary {{
                background: {PANEL_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
            }}
            QPushButton#secondary:hover {{ background: {BORDER_COLOR}; }}
            QLabel {{ color: {TEXT_COLOR}; font-size: 13px; }}
            QLabel#heading {{ font-size: 13px; font-weight: bold; color: #202124; }}
            QSpinBox {{
                background: {PANEL_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 2px 4px;
                border-radius: 3px;
            }}
            QScrollBar:vertical {{
                background: {BG_COLOR}; width: 8px; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR}; border-radius: 4px;
            }}
            QStatusBar {{ background: {PANEL_COLOR}; color: {TEXT_COLOR}; font-size: 11px; }}
            QSplitter::handle {{ background: {BORDER_COLOR}; }}
        """)

        self._df: pd.DataFrame | None = None
        self._time_ms_full: np.ndarray | None = None
        self._check_states: dict[int, bool] = {}
        # Whether in merged view currently? 
        self._merged_view = False

        self._build_ui()
        self._status("No file loaded — use Open to get started.")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setStyleSheet(f"background: {PANEL_COLOR}; border-bottom: 1px solid {BORDER_COLOR}; spacing: 6px; padding: 4px;")
        self.addToolBar(toolbar)

        open_action = QAction("📂  Open Excel…", self)
        open_action.triggered.connect(self._open_file)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        sel_all_btn = QPushButton("Select All")
        sel_all_btn.setObjectName("secondary")
        sel_all_btn.clicked.connect(lambda: self._set_all_checks(True))
        toolbar.addWidget(sel_all_btn)

        sel_none_btn = QPushButton("Select None")
        sel_none_btn.setObjectName("secondary")
        sel_none_btn.clicked.connect(lambda: self._set_all_checks(False))
        toolbar.addWidget(sel_none_btn)

        #Adding Merge and Back buttons in the UI 
        toolbar.addSeparator()

        self._merge_btn = QPushButton("⬡ Merge")
        self._merge_btn.setObjectName("secondary")
        self._merge_btn.clicked.connect(self._show_merged)
        toolbar.addWidget(self._merge_btn)

        #Input/Output curve button
        self._io_btn = QPushButton("⬡ Input/Output Graph")
        self._io_btn.setObjectName("secondary")
        self._io_btn.clicked.connect(self._show_io_curve)
        toolbar.addWidget(self._io_btn)

        self._back_btn = QPushButton("← Back")
        self._back_btn.setObjectName("secondary")
        self._back_btn.clicked.connect(self._show_stacked)
        self._back_btn.setVisible(False)
        toolbar.addWidget(self._back_btn)


        #Channel filter 
        toolbar.addSeparator()

        channel_label = QLabel("⬡ Filter by Channel:")
        toolbar.addWidget(channel_label)

        self._channel_combo = QComboBox()

        self._channel_combo.addItems([
            "All Channels",
            "Near-Field VsEP",
            "Acceleration",
            "Microphone",
        ])

        toolbar.addWidget(self._channel_combo)

        self._channel_combo.currentTextChanged.connect(
            self._on_channel_changed
        )


        # Central splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # Top: data table
        top_frame = QFrame()
        top_layout = QVBoxLayout(top_frame)
        top_layout.setContentsMargins(8, 8, 8, 4)
        top_layout.setSpacing(4)

        tbl_label = QLabel("Data Table")
        tbl_label.setObjectName("heading")
        top_layout.addWidget(tbl_label)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.itemChanged.connect(self._on_check_changed)
        top_layout.addWidget(self._table)

        splitter.addWidget(top_frame)

        # Bottom: waveform panel
        bottom_frame = QFrame()
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(8, 4, 8, 8)
        bottom_layout.setSpacing(4)

        self._wave_label = QLabel("Waveforms")
        self._wave_label.setObjectName("heading")
        bottom_layout.addWidget(self._wave_label)

        self._wave_panel = WaveformPanel()
        bottom_layout.addWidget(self._wave_panel)

        splitter.addWidget(bottom_frame)
        splitter.setSizes([480, 300])

        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

    # ── File loading ──────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Excel File", "",
            "Excel Files (*.xlsx *.xls *.xlsm *.xlsb);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if not path:
            return
        try:
            # Two header rows — combine them into "Header / Subheader" column names
            self._df = pd.read_excel(path, header=[0, 1])
            # Flatten multi-index columns to readable strings
            self._df.columns = [
                " / ".join(str(s) for s in col if not str(s).startswith("Unnamed"))
                or f"Col{i}"
                for i, col in enumerate(self._df.columns)
            ]
            # Read the full time row from the waveform region.
            # Keep the original column positions intact. We do not remove NaNs
            # here because X and Y must stay positionally aligned by Excel column.

            raw = pd.read_excel(path, header=None)
            self._time_ms_full = pd.to_numeric(
               raw.iloc[0, PCM_COL_START:],
               errors="coerce"
            ).to_numpy(dtype=float)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not read file:\n{e}")
            return

        self._check_states = {i: False for i in range(len(self._df))}
        self._populate_table()
        n_pcm = max(0, len(self._df.columns) - PCM_COL_START)
        self._status(
            f"Loaded {len(self._df)} rows — "
            f"{len(self._df.columns[:9])} metadata cols, {n_pcm} PCM sample cols"
        )

    # ── Table population ──────────────────────────────────────────────────────

    def _populate_table(self):
        if self._df is None:
            return

        df = self._df
        meta_cols = list(df.columns[META_COL_SLICE])
        meta_cols.pop(1)

        #Adding more Columns to GUI, for Cochlear and Vestibular PtP
        #Removed Vestibular Min Max, Cochlear Min Max, Global Min Max and PCM samples Columns. 
        col_headers = ["✓"] + meta_cols + ["Vestibular PtP","Cochlear PtP"]
        self._table.blockSignals(True)
        self._table.clearContents()
        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(col_headers))
        self._table.setHorizontalHeaderLabels([str(c) for c in col_headers])

        for row_idx, (_, row) in enumerate(df.iterrows()):
            # Checkbox cell
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(
                Qt.CheckState.Checked if self._check_states.get(row_idx) else Qt.CheckState.Unchecked
            )
            chk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_item.setBackground(QColor(COLUMN_COLORS["✓"]))
            self._table.setItem(row_idx, 0, chk_item)

            # Metadata cells
            for col_offset, col in enumerate(meta_cols):
                item = QTableWidgetItem(str(row[col]))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setBackground(QColor(NEUTRAL_METADATA_COLOR))
                self._table.setItem(row_idx, col_offset + 1, item)

            # Extract PtP measurements for display
            vest_ptp = row.iloc[11]
            coch_ptp = row.iloc[14]

            measurement_values = [
            (vest_ptp,VESTIBULAR_COLOR),
            (coch_ptp,COCHLEAR_COLOR),
            ]

            #Optional display depending on NA value logic 
            #Need more understanding??
            start_col = 1 + len(meta_cols)

            for offset, (value,column_color) in enumerate(measurement_values):
                item = QTableWidgetItem(format_optional_value(value))
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled |
                    Qt.ItemFlag.ItemIsSelectable
                )

                #Apply Asiigned Column Colour 
                item.setBackground(
                    QColor(column_color)
                )

                if pd.isna(value):
                    item.setForeground(QColor("#666b78"))
            #else:
                #item.setForeground(QColor("#c8cdd8"))
            
            #Make sure the following part is in the for loop
            #indentation sensitive
                self._table.setItem(
                row_idx,
                start_col + offset,
                item
                )

        # Column widths
        self._table.setColumnWidth(0, 36)
        for i in range(1, len(col_headers)):
            self._table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
            )

        self._table.blockSignals(False)
        self._rebuild_waveforms()

    # ── Checkbox handling ─────────────────────────────────────────────────────

    def _on_check_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        row_idx = item.row()
        self._check_states[row_idx] = item.checkState() == Qt.CheckState.Checked
        self._rebuild_waveforms()
        selected = sum(self._check_states.values())
        self._status(f"{selected} row(s) selected")

    def _set_all_checks(self, state: bool):
        if self._df is None:
            return
        self._table.blockSignals(True)
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
            self._check_states[row_idx] = state
        self._table.blockSignals(False)
        self._rebuild_waveforms()

    # ── Waveform rebuilding ───────────────────────────────────────────────────
    def _get_waveform(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a validated (time_ms, pcm) pair for one recording.

        A recording may end before the full Excel time template ends. Trailing
        blank PCM cells are therefore treated as the end of that recording.
        Missing values inside the active waveform span are treated as invalid
        instead of being silently removed.
        """
        if self._df is None:
           raise ValueError("No Excel file is loaded.")

        if self._time_ms_full is None:
           raise ValueError("No time-axis data is available.")

        pcm_raw = pd.to_numeric(
            self._df.iloc[row_idx, PCM_COL_START:],
            errors="coerce"
        ).to_numpy(dtype=float)

        time_raw = self._time_ms_full

        # The full X and Y regions come from the same Excel columns.
        if len(time_raw) != len(pcm_raw):
            raise ValueError(
               f"Time/sample column mismatch: "
                f"{len(time_raw)} time values vs {len(pcm_raw)} PCM cells."
            )

        valid_pcm = np.isfinite(pcm_raw)

        if not valid_pcm.any():
           raise ValueError("This recording contains no PCM samples.")

       # Trailing blanks are allowed and mark the end of a shorter recording.
        last_valid_index = int(np.flatnonzero(valid_pcm)[-1])

        time_ms = time_raw[:last_valid_index + 1]
        pcm = pcm_raw[:last_valid_index + 1]

        # An internal gap would break the time-to-amplitude correspondence.
        if not np.all(np.isfinite(pcm)):
            raise ValueError(
                "The waveform contains missing PCM samples inside the recording."
            )

        if not np.all(np.isfinite(time_ms)):
           raise ValueError(
                "The corresponding time axis contains missing values."
            )

        # Hard requirement: each amplitude sample must have one time value.
        if len(time_ms) != len(pcm):
            raise ValueError(
                f"X/Y mismatch: {len(time_ms)} time points "
               f"vs {len(pcm)} PCM samples."
            )

        if len(time_ms) > 1 and not np.all(np.diff(time_ms) > 0):
           raise ValueError("The time axis is not strictly increasing.")

        return time_ms, pcm


    #Channel filtering logic
    def _filter_channel_window(
        self,
        time_ms: np.ndarray,
        pcm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        channel = self._channel_combo.currentText()

        # Show the full waveform
        if channel == "All Channels":
            return time_ms, pcm

        start_ms, end_ms = CHANNEL_WINDOWS[channel]

        if channel == "Microphone":
            mask = (
                (time_ms >= start_ms) &
                (time_ms <= end_ms)
            )
        else:
            mask = (
                (time_ms >= start_ms) &
                (time_ms < end_ms)
            )

        filtered_time = time_ms[mask]
        filtered_pcm = pcm[mask]

        if len(filtered_time) == 0:
            raise ValueError(
                f"No samples found in the "
                f"{start_ms:g}–{end_ms:g} ms window."
            )

        return filtered_time, filtered_pcm



    #Channel selection changed
    def _on_channel_changed(self, channel_name: str):
        if self._df is None:
            return

        self._rebuild_waveforms()



   

    def _row_label(self, row_idx: int) -> str:
        if self._df is None:
            return f"Row {row_idx + 1}"
        meta  = self._df.iloc[row_idx, META_COL_SLICE]
        parts = [str(v) for v in meta.values if str(v).strip() and str(v) != "nan"]
        return "  ·  ".join(parts) if parts else f"Row {row_idx + 1}"

    def _rebuild_waveforms(self):
        if self._df is None:
            return
        selected_indices = [i for i, v in self._check_states.items() if v]
        if not selected_indices:
            self._wave_panel.rebuild([])
            return

        rows_data = []
        errors = []

        for i, row_idx in enumerate(selected_indices):
            color = WAVEFORM_COLORS[i % len(WAVEFORM_COLORS)]
            label = self._row_label(row_idx)

            try:
                time_ms, pcm = self._get_waveform(row_idx)

                time_ms, pcm = self._filter_channel_window(
                    time_ms, 
                    pcm
                )
            except ValueError as e:
                errors.append(f"{label}:{e}")
                continue

            rows_data.append({
                "label": label, 
                "color": color, 
                "time_ms": time_ms,
                "pcm":pcm
            })
        self._wave_panel.rebuild(rows_data)
    
    # Update waveform panel title
        channel = self._channel_combo.currentText()
        
        if channel == "All Channels":
            self._wave_label.setText(
                "Waveforms — All Channels"
            )

        else:
            start_ms, end_ms = CHANNEL_WINDOWS[channel]

            self._wave_label.setText(
                f"Waveforms — {channel} "
                f"({start_ms:g}–{end_ms:g} ms)"
            )


        if errors:
          self._status(
              f"Skipped {len(errors)} invalid waveform(s). "
              f"First issue: {errors[0]}"
              )
    # ── Input / Output Curve ──────────────────────────────────────────────────────

    def _show_io_curve(self):

        if self._df is None:
            self._status(
                "Load an Excel file before generating an I/O curve."
            )
            return

        result = build_io_curve_from_table(
            parent=self,
            table=self._table,
            check_states=self._check_states,

            panel_color=PANEL_COLOR,
            border_color=BORDER_COLOR,
            text_color=TEXT_COLOR,

            cochlear_plot_color=COCHLEAR_PLOT_COLOR,
            vestibular_plot_color=VESTIBULAR_PLOT_COLOR,

            attach_hover=attach_hover_readout,
        )

        if result is None:
            return

        self._wave_panel.show_plot_widget(
            result.plot
        )

        self._wave_label.setText(
            result.title
        )

        self._merge_btn.setVisible(False)
        self._io_btn.setVisible(False)
        self._back_btn.setVisible(True)

        self._status(
            f"I/O curve generated from "
            f"{result.point_count} recording(s)."
        )
    
    # ── Merged Waveform ────────────────────────────────────────────────────────────
    def _show_merged(self):
        if self._df is None:
            return
        selected = [i for i, v in self._check_states.items() if v]
    
        if not selected:
            self._status("No rows selected.")
            return
    
        if len(selected) > 40:
            self._status("Max 40 waveforms for merge — deselect some first.")
            return
   
        self._merged_view = True
        rows_data = []
        errors = []
        for i, row_idx in enumerate(selected):
            color = WAVEFORM_COLORS[i % len(WAVEFORM_COLORS)]
            label = self._row_label(row_idx)

            try:
                time_ms, pcm = self._get_waveform(row_idx)
            except ValueError as e:
                errors.append(f"{label}:{e}")
                continue

            rows_data.append({
                "label": label, 
                "color": color, 
                "time_ms": time_ms,
                "pcm":pcm
            })

        if not rows_data: 
            self._merged_view = False
            self.status(
                "None of the selected recordings contain valid waveform data."
            )
            return

        self._wave_panel.show_merged(rows_data)
        self._wave_label.setText(
            f"Waveforms — Merged ({len(rows_data)} overlaid)"
        )
        self._merge_btn.setVisible(False)
        self._back_btn.setVisible(True)

        if errors: 
            self._status(
                f"Showing {len(rows_data)} waveform(s); "
                f"skipped {len(errors)} invalid waveform(s). "
                f"First issue: {errors[0]}"
            )
        else:
            self._status(f"Showing {len(rows_data)} waveform(s) overlaid.")


    # ── Stacked Waveform ────────────────────────────────────────────────────────────
    def _show_stacked(self):
        self._merged_view = False
        self._rebuild_waveforms()
        self._merge_btn.setVisible(True)
        self._io_btn.setVisible(True)
        self._back_btn.setVisible(False)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.statusBar().showMessage(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    pg.setConfigOptions(antialias=True, foreground=TEXT_COLOR, background=BG_COLOR)

    app = QApplication(sys.argv)
    app.setApplicationName("PCM Waveform Viewer")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
