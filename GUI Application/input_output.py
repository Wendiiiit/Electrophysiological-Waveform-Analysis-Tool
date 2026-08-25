#The feature currently:
#- uses only checked rows from the Data Table;
#- uses "dB atten." as the X variable;
#- lets the user choose Cochlear PtP or Vestibular PtP as Y;
#- lets the user restrict the included dB range;
#- supports automatic or manual Y-axis display limits;
#- sorts observations by dB before connecting points;
#- keeps repeated dB observations as separate points (no automatic averaging).


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QMessageBox,
    QTableWidget,
    QWidget,
)


DB_HEADER = "dB atten."
COCHLEAR_HEADER = "Cochlear PtP"
VESTIBULAR_HEADER = "Vestibular PtP"
BOTH_HEADER = "Both"


@dataclass
class IOCurveResult:
    """Result returned to the main GUI after a curve is successfully built."""

    plot: pg.PlotWidget
    title: str
    point_count: int
    measurement_name: str


@dataclass
class IOCurveSettings:
    """User-selected settings for one I/O Graph."""

    measurement_name: str
    db_min: float
    db_max: float
    auto_y: bool
    y_min: float | None
    y_max: float | None


def _find_table_column(table: QTableWidget, header_name: str) -> int | None:
    """Return the visible table-column index whose header matches header_name."""

    for column in range(table.columnCount()):
        header_item = table.horizontalHeaderItem(column)
        if header_item is not None and header_item.text() == header_name:
            return column
    return None


def _selected_rows(check_states: dict[int, bool]) -> list[int]:
    """Return row indices currently checked by the user."""

    return [row_idx for row_idx, checked in check_states.items() if checked]


def _selected_db_values(
    table: QTableWidget,
    selected_rows: list[int],
    db_col: int,
) -> list[float]:
    """Collect finite dB values from the selected rows."""

    values: list[float] = []

    for row_idx in selected_rows:
        item = table.item(row_idx, db_col)
        if item is None:
            continue

        try:
            value = float(item.text())
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return values


class IOCurveDialog(QDialog):
    """Dialog used to configure an input–output curve."""

    def __init__(self, parent: QWidget, db_values: list[float]):
        super().__init__(parent)

        self.setWindowTitle("Generate I/O Graph")
        self.setMinimumWidth(380)

        layout = QFormLayout(self)

        self.measurement_combo = QComboBox()
        self.measurement_combo.addItems([
            COCHLEAR_HEADER,
            VESTIBULAR_HEADER,
            BOTH_HEADER,
        ])
        layout.addRow("Measurement Type:", self.measurement_combo)

        self.db_min = QDoubleSpinBox()
        self.db_min.setRange(-1_000_000.0, 1_000_000.0)
        self.db_min.setDecimals(3)
        self.db_min.setValue(min(db_values))
        layout.addRow("Minimum dB attenuation:", self.db_min)

        self.db_max = QDoubleSpinBox()
        self.db_max.setRange(-1_000_000.0, 1_000_000.0)
        self.db_max.setDecimals(3)
        self.db_max.setValue(max(db_values))
        layout.addRow("Maximum dB attenuation:", self.db_max)

        self.auto_y = QCheckBox("Auto-scale Y-axis")
        self.auto_y.setChecked(True)
        layout.addRow("", self.auto_y)

        self.y_min = QDoubleSpinBox()
        self.y_min.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.y_min.setDecimals(3)
        self.y_min.setValue(0.0)
        self.y_min.setEnabled(False)
        layout.addRow("PtP minimum:", self.y_min)

        self.y_max = QDoubleSpinBox()
        self.y_max.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.y_max.setDecimals(3)
        self.y_max.setValue(500.0)
        self.y_max.setEnabled(False)
        layout.addRow("PtP maximum:", self.y_max)

        self.auto_y.toggled.connect(self._toggle_manual_y)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Generate")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _toggle_manual_y(self, auto_scale: bool):
        self.y_min.setEnabled(not auto_scale)
        self.y_max.setEnabled(not auto_scale)

    def settings(self) -> IOCurveSettings:
        auto_y = self.auto_y.isChecked()
        return IOCurveSettings(
            measurement_name=self.measurement_combo.currentText(),
            db_min=self.db_min.value(),
            db_max=self.db_max.value(),
            auto_y=auto_y,
            y_min=None if auto_y else self.y_min.value(),
            y_max=None if auto_y else self.y_max.value(),
        )


def _collect_io_data(
    table: QTableWidget,
    selected_rows: list[int],
    db_col: int,
    ptp_col: int,
    db_min: float,
    db_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract valid (dB, PtP) pairs from checked rows and sort by dB."""

    x_values: list[float] = []
    y_values: list[float] = []

    for row_idx in selected_rows:
        db_item = table.item(row_idx, db_col)
        ptp_item = table.item(row_idx, ptp_col)

        if db_item is None or ptp_item is None:
            continue

        try:
            db_value = float(db_item.text())
            ptp_value = float(ptp_item.text())
        except (TypeError, ValueError):
            # Also skips display placeholders such as "—".
            continue

        if not (np.isfinite(db_value) and np.isfinite(ptp_value)):
            continue

        if not (db_min <= db_value <= db_max):
            continue

        x_values.append(db_value)
        y_values.append(ptp_value)

    if len(x_values) < 2:
        raise ValueError(
            "At least two valid selected recordings are needed "
            "to generate an I/O Graph."
        )

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    # Connect points in increasing dB order rather than spreadsheet row order.
    order = np.argsort(x, kind="stable")
    return x[order], y[order]

def _build_plot(
    *,
    x1: np.ndarray,
    y1: np.ndarray,
    label1: str,
    color1: str,
    settings: IOCurveSettings,
    panel_color: str,
    border_color: str,
    text_color: str,
    attach_hover: Callable[[pg.PlotWidget], None] | None = None,
    x2: np.ndarray | None = None,
    y2: np.ndarray | None = None,
    label2: str | None = None,
    color2: str | None = None,
) -> pg.PlotWidget:
    """Create and style the pyqtgraph I/O plot."""

    plot = pg.PlotWidget(background=panel_color)

    if settings.measurement_name == BOTH_HEADER:
        title_text = "I/O Graph — Cochlear and Vestibular PtP"
        y_label_text = "PtP"
    else:
        title_text = f"I/O Graph — {settings.measurement_name}"
        y_label_text = settings.measurement_name

    plot.setTitle(
        title_text,
        color=text_color,
        size="12pt",
    )

    bottom_axis = plot.getAxis("bottom")
    left_axis = plot.getAxis("left")

    bottom_axis.setLabel("dB attenuation", color=text_color)
    left_axis.setLabel(y_label_text, color=text_color)

    for axis_name in ("bottom", "left", "top", "right"):
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(border_color))
        axis.setTextPen(pg.mkPen(text_color))

    plot.showGrid(x=True, y=True, alpha=0.15)

    # Add legend only if plotting two series
    if x2 is not None and y2 is not None:
        plot.addLegend()

    # First series
    plot.plot(
        x1,
        y1,
        pen=pg.mkPen(color1, width=2),
        symbol="o",
        symbolSize=8,
        symbolBrush=pg.mkBrush(color1),
        symbolPen=pg.mkPen(color1),
        antialias=True,
        name=label1,
    )

    # Optional second series
    if (
        x2 is not None
        and y2 is not None
        and label2 is not None
        and color2 is not None
    ):
        plot.plot(
            x2,
            y2,
            pen=pg.mkPen(color2, width=2),
            symbol="o",
            symbolSize=8,
            symbolBrush=pg.mkBrush(color2),
            symbolPen=pg.mkPen(color2),
            antialias=True,
            name=label2,
        )

    if settings.y_min is not None and settings.y_max is not None:
        plot.setYRange(
            settings.y_min,
            settings.y_max,
            padding=0,
        )

    if attach_hover is not None:
        attach_hover(plot)

    return plot


def build_io_curve_from_table(
    *,
    parent: QWidget,
    table: QTableWidget,
    check_states: dict[int, bool],
    panel_color: str,
    border_color: str,
    text_color: str,
    cochlear_plot_color: str,
    vestibular_plot_color: str,
    attach_hover: Callable[[pg.PlotWidget], None] | None = None,
) -> IOCurveResult | None:
    """
    Run the complete I/O-curve workflow.

    Returns IOCurveResult when a curve is successfully generated.
    Returns None when the user cancels or the request cannot be completed.
    """

    selected_rows = _selected_rows(check_states)

    if not selected_rows:
        QMessageBox.information(
            parent,
            "I/O Graph",
            "Select the recordings you want to include first.",
        )
        return None

    db_col = _find_table_column(table, DB_HEADER)
    if db_col is None:
        QMessageBox.warning(
            parent,
            "I/O Graph",
            f"Could not find the '{DB_HEADER}' column.",
        )
        return None

    db_values = _selected_db_values(table, selected_rows, db_col)
    if not db_values:
        QMessageBox.warning(
            parent,
            "I/O Graph",
            "The selected rows contain no valid dB values.",
        )
        return None

    dialog = IOCurveDialog(parent=parent, db_values=db_values)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    settings = dialog.settings()

    if settings.db_min > settings.db_max:
        QMessageBox.warning(
            parent,
            "I/O Graph",
            "Minimum dB must be smaller than or equal to maximum dB.",
        )
        return None

    if (
        settings.y_min is not None
        and settings.y_max is not None
        and settings.y_min >= settings.y_max
    ):
        QMessageBox.warning(
            parent,
            "I/O Graph",
            "Y-axis minimum must be smaller than Y-axis maximum.",
        )
        return None

    # ── BOTH series mode ─────────────────────────────────────────────
    if settings.measurement_name == BOTH_HEADER:

        coch_col = _find_table_column(table, COCHLEAR_HEADER)
        vest_col = _find_table_column(table, VESTIBULAR_HEADER)

        if coch_col is None or vest_col is None:
            QMessageBox.warning(
                parent,
                "I/O Graph",
                "Could not find both Cochlear PtP and Vestibular PtP columns.",
            )
            return None

        try:
            x_coch, y_coch = _collect_io_data(
                table=table,
                selected_rows=selected_rows,
                db_col=db_col,
                ptp_col=coch_col,
                db_min=settings.db_min,
                db_max=settings.db_max,
            )
        except ValueError:
            x_coch = np.array([], dtype=float)
            y_coch = np.array([], dtype=float)

        try:
            x_vest, y_vest = _collect_io_data(
                table=table,
                selected_rows=selected_rows,
                db_col=db_col,
                ptp_col=vest_col,
                db_min=settings.db_min,
                db_max=settings.db_max,
            )
        except ValueError:
            x_vest = np.array([], dtype=float)
            y_vest = np.array([], dtype=float)

        if len(x_coch) == 0 and len(x_vest) == 0:
            QMessageBox.warning(
                parent,
                "I/O Graph",
                "No valid Cochlear or Vestibular PtP data could be plotted.",
            )
            return None

        if len(x_coch) == 0:
            QMessageBox.warning(
                parent,
                "I/O Graph",
                "No valid Cochlear PtP data found. Plotting Vestibular only is recommended.",
            )

        if len(x_vest) == 0:
            QMessageBox.warning(
                parent,
                "I/O Graph",
                "No valid Vestibular PtP data found. Plotting Cochlear only is recommended.",
            )

        plot = _build_plot(
            x1=x_coch,
            y1=y_coch,
            label1=COCHLEAR_HEADER,
            color1=cochlear_plot_color,
            x2=x_vest if len(x_vest) > 0 else None,
            y2=y_vest if len(y_vest) > 0 else None,
            label2=VESTIBULAR_HEADER if len(x_vest) > 0 else None,
            color2=vestibular_plot_color if len(x_vest) > 0 else None,
            settings=settings,
            panel_color=panel_color,
            border_color=border_color,
            text_color=text_color,
            attach_hover=attach_hover,
    )

# ── SINGLE series mode ───────────────────────────────────────────

    ptp_col = _find_table_column(
        table,
        settings.measurement_name,
    )

    if ptp_col is None:
        QMessageBox.warning(
            parent,
            "I/O Graph",
            f"Could not find the '{settings.measurement_name}' column.",
        )
        return None


    try:
        x, y = _collect_io_data(
            table=table,
            selected_rows=selected_rows,
            db_col=db_col,
            ptp_col=ptp_col,
            db_min=settings.db_min,
            db_max=settings.db_max,
        )

    except ValueError as exc:
        QMessageBox.warning(
            parent,
            "I/O Graph",
            str(exc),
        )
        return None


    if settings.measurement_name == VESTIBULAR_HEADER:
        line_color = vestibular_plot_color
    else:
        line_color = cochlear_plot_color


    plot = _build_plot(
        x1=x,
        y1=y,
        label1=settings.measurement_name,
        color1=line_color,
        settings=settings,
        panel_color=panel_color,
        border_color=border_color,
        text_color=text_color,
        attach_hover=attach_hover,
    )


    return IOCurveResult(
        plot=plot,
        title=f"I/O Graph — {settings.measurement_name}",
        point_count=len(x),
        measurement_name=settings.measurement_name,
    )
        


