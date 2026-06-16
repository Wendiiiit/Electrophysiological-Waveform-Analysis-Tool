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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QLabel, QFileDialog, QScrollArea, QFrame,
    QCheckBox, QMessageBox, QStatusBar, QToolBar,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QAction, QIcon

import pyqtgraph as pg


# ── Colour palette ────────────────────────────────────────────────────────────
WAVEFORM_COLORS = [
    "#4e9af1", "#f16c4e", "#4ef18a", "#f1d44e",
    "#c44ef1", "#4ef1e8", "#f14e8a", "#a8f14e",
]
MERGED_COLOR   = "#ffffff"
BG_COLOR       = "#1a1d23"
PANEL_COLOR    = "#22252e"
BORDER_COLOR   = "#2e3240"
TEXT_COLOR      = "#c8cdd8"
ACCENT_COLOR   = "#4e9af1"


# ── Layout constants ──────────────────────────────────────────────────────────
# Columns A–I (0-based: 0–8) are metadata labels
# Columns J–O (9–14) are skipped
# Columns P+ (15 onwards) are PCM data

META_COL_SLICE = slice(0, 9)    # A–I
PCM_COL_START  = 15             # P (0-based)


def merge_pcm(arrays: list[np.ndarray]) -> np.ndarray:
    """Sum multiple PCM arrays (audio mix), truncating to the shortest length."""
    if not arrays:
        return np.array([])
    min_len = min(len(a) for a in arrays)
    stacked = np.vstack([a[:min_len].astype(float) for a in arrays])
    return stacked.sum(axis=0)

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
        self.plot(time_ms, pcm, pen=pen, antialias=True)

        # Horizontal zero line
        self.addLine(y=0, pen=pg.mkPen(color=BORDER_COLOR, style=Qt.PenStyle.DashLine))

        # Disable all mouse interaction — scroll wheel scrolls the list instead
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()


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
            self.add_waveform(row["label"], row["color"], row["pcm"], self.time_ms)
    
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
            plot.plot(self.time_ms, row["pcm"], pen=pen, antialias=True)

        self._layout.insertWidget(self._layout.count() - 1, plot)
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
                gridline-color: {BORDER_COLOR};
                border: 1px solid {BORDER_COLOR};
                color: {TEXT_COLOR};
                font-size: 12px;
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
            QLabel {{ color: {TEXT_COLOR}; font-size: 12px; }}
            QLabel#heading {{ font-size: 13px; font-weight: bold; color: #ffffff; }}
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

        self._merge_btn = QPushButton("⬡  Merge")
        self._merge_btn.clicked.connect(self._show_merged)
        toolbar.addWidget(self._merge_btn)

        self._back_btn = QPushButton("← Back")
        self._back_btn.setObjectName("secondary")
        self._back_btn.clicked.connect(self._show_stacked)
        self._back_btn.setVisible(False)
        toolbar.addWidget(self._back_btn)







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
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{ alternate-background-color: #1e2128; }}
        """)
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
            # Also grab the amplitude data headings
            raw = pd.read_excel(path, header=None)
            raw_times = raw.iloc[0, PCM_COL_START:].values.astype(float) # 'iloc[1:4, 15:20]' means 'p2:t4' in Excel parlance
            mask = np.isfinite(raw_times) # only return cells that have a value
            self._wave_panel.time_ms = raw_times[mask]

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
        pcm_cols  = list(df.columns[PCM_COL_START:])

        col_headers = ["✓"] + meta_cols + ["PCM Samples", "Min", "Max"]
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
            self._table.setItem(row_idx, 0, chk_item)

            # Metadata cells
            for col_offset, col in enumerate(meta_cols):
                item = QTableWidgetItem(str(row[col]))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._table.setItem(row_idx, col_offset + 1, item)

            # PCM summary
            pcm_data = row[pcm_cols].values.astype(float)
            valid    = pcm_data[~np.isnan(pcm_data)]
            n_samp   = QTableWidgetItem(str(len(valid)))
            n_min    = QTableWidgetItem(f"{valid.min():.2f}" if len(valid) else "—")
            n_max    = QTableWidgetItem(f"{valid.max():.2f}" if len(valid) else "—")
            for item in (n_samp, n_min, n_max):
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setForeground(QColor("#888ea0"))
            self._table.setItem(row_idx, len(col_headers) - 3, n_samp)
            self._table.setItem(row_idx, len(col_headers) - 2, n_min)
            self._table.setItem(row_idx, len(col_headers) - 1, n_max)

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

    def _get_pcm(self, row_idx: int) -> np.ndarray:
        row   = self._df.iloc[row_idx, PCM_COL_START:]
        data  = row.values.astype(float)
        valid = data[~np.isnan(data)]
        return valid

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
        for i, row_idx in enumerate(selected_indices):
            pcm   = self._get_pcm(row_idx)
            color = WAVEFORM_COLORS[i % len(WAVEFORM_COLORS)]
            label = self._row_label(row_idx)
            rows_data.append({"label": label, "color": color, "pcm": pcm})
        
        self._wave_panel.rebuild(rows_data)
        self._wave_label.setText("Waveforms")
    
    # ── Merged Waveform ────────────────────────────────────────────────────────────
    def _show_merged(self):
        if self._df is None:
            return
        selected = [i for i, v in self._check_states.items() if v]
    
        if not selected:
            self._status("No rows selected.")
            return
    
        if len(selected) > 10:
            self._status("Max 10 waveforms for merge — deselect some first.")
            return
   
        self._merged_view = True
        rows_data = []
        for i, row_idx in enumerate(selected):
            pcm   = self._get_pcm(row_idx)
            color = WAVEFORM_COLORS[i % len(WAVEFORM_COLORS)]
            label = self._row_label(row_idx)
            rows_data.append({"label": label, "color": color, "pcm": pcm})
        self._wave_panel.show_merged(rows_data)
        self._wave_label.setText(f"Waveforms — Merged ({len(selected)} overlaid)")
        self._merge_btn.setVisible(False)
        self._back_btn.setVisible(True)
        self._status(f"Showing {len(selected)} waveform(s) overlaid.")

    # ── Stacked Waveform ────────────────────────────────────────────────────────────
    def _show_stacked(self):
        self._merged_view = False
        self._rebuild_waveforms()
        self._merge_btn.setVisible(True)
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
