"""
Unit tests for input_output.py (I/O Graph feature).

Run with:
    pytest test_input_output.py -v

On a machine with no display (e.g. CI), run headless:
    QT_QPA_PLATFORM=offscreen pytest test_input_output.py -v

Only pytest, numpy, PyQt6 and pyqtgraph are required. Modal dialogs and
message boxes are replaced with fakes so the tests never block.
"""

from __future__ import annotations

import os
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog, QTableWidget, QTableWidgetItem, QWidget

import input_output as io_mod
from input_output import (
    BOTH_HEADER,
    COCHLEAR_HEADER,
    DB_HEADER,
    VESTIBULAR_HEADER,
    IOCurveDialog,
    IOCurveSettings,
    _build_plot,
    _collect_io_data,
    _find_table_column,
    _selected_db_values,
    _selected_rows,
    build_io_curve_from_table,
    export_io_graph,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────

PANEL = "#ffffff"
BORDER = "#888888"
TEXT = "#000000"
COCH_COLOR = "#1f77b4"
VEST_COLOR = "#d62728"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def parent(qapp):
    widget = QWidget()
    yield widget
    widget.deleteLater()


def make_table(headers: list[str], rows: list[list[str | None]]) -> QTableWidget:
    """Build a QTableWidget. A cell value of None leaves the cell empty (no item)."""
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value is not None:
                table.setItem(r, c, QTableWidgetItem(value))
    return table


STANDARD_HEADERS = ["File", DB_HEADER, COCHLEAR_HEADER, VESTIBULAR_HEADER]


@pytest.fixture
def standard_table(qapp):
    # Rows deliberately out of dB order to test sorting.
    return make_table(
        STANDARD_HEADERS,
        [
            ["a.pcm", "80", "120.5", "30.0"],
            ["b.pcm", "60", "80.0", "20.0"],
            ["c.pcm", "100", "200.0", "45.0"],
            ["d.pcm", "70", "95.0", "—"],
        ],
    )


def all_checked(n: int) -> dict[int, bool]:
    return {i: True for i in range(n)}


def make_settings(**overrides) -> IOCurveSettings:
    defaults = dict(
        measurement_name=COCHLEAR_HEADER,
        db_min=-1e6,
        db_max=1e6,
        auto_y=True,
        y_min=None,
        y_max=None,
    )
    defaults.update(overrides)
    return IOCurveSettings(**defaults)


class MessageRecorder:
    """Stand-in for QMessageBox that records calls instead of showing dialogs."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def information(self, parent, title, text):
        self.calls.append(("information", title, text))

    def warning(self, parent, title, text):
        self.calls.append(("warning", title, text))


@pytest.fixture
def messages(monkeypatch):
    recorder = MessageRecorder()
    monkeypatch.setattr(io_mod, "QMessageBox", recorder)
    return recorder


@pytest.fixture
def fake_dialog(monkeypatch):
    """
    Replace IOCurveDialog with a non-modal fake.

    Usage: fake_dialog(settings, accepted=True). The created fake records the
    db_values it was constructed with in fake_dialog.received_db_values.
    """

    state = {}

    def configure(settings: IOCurveSettings | None = None, accepted: bool = True):
        state["settings"] = settings or make_settings()
        state["accepted"] = accepted

        class FakeDialog:
            def __init__(self, parent, db_values):
                configure.received_db_values = list(db_values)

            def exec(self):
                return (
                    QDialog.DialogCode.Accepted
                    if state["accepted"]
                    else QDialog.DialogCode.Rejected
                )

            def settings(self):
                return state["settings"]

        monkeypatch.setattr(io_mod, "IOCurveDialog", FakeDialog)
        return configure

    configure.received_db_values = None
    return configure


def run_workflow(parent, table, check_states, attach_hover=None):
    return build_io_curve_from_table(
        parent=parent,
        table=table,
        check_states=check_states,
        panel_color=PANEL,
        border_color=BORDER,
        text_color=TEXT,
        cochlear_plot_color=COCH_COLOR,
        vestibular_plot_color=VEST_COLOR,
        attach_hover=attach_hover,
    )


def data_items(plot):
    return plot.plotItem.listDataItems()


# ─────────────────────────────────────────────────────────────────────
# _find_table_column
# ─────────────────────────────────────────────────────────────────────

class TestFindTableColumn:
    def test_finds_existing_header(self, standard_table):
        assert _find_table_column(standard_table, DB_HEADER) == 1
        assert _find_table_column(standard_table, VESTIBULAR_HEADER) == 3

    def test_missing_header_returns_none(self, standard_table):
        assert _find_table_column(standard_table, "Nonexistent") is None

    def test_match_is_exact(self, standard_table):
        assert _find_table_column(standard_table, "dB atten") is None
        assert _find_table_column(standard_table, "cochlear ptp") is None

    def test_empty_table(self, qapp):
        assert _find_table_column(QTableWidget(), DB_HEADER) is None

    def test_returns_first_match_for_duplicate_headers(self, qapp):
        table = make_table([DB_HEADER, DB_HEADER], [])
        assert _find_table_column(table, DB_HEADER) == 0

    def test_column_without_header_item_is_skipped(self, qapp):
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderItem(1, QTableWidgetItem(DB_HEADER))
        assert _find_table_column(table, DB_HEADER) == 1


# ─────────────────────────────────────────────────────────────────────
# _selected_rows
# ─────────────────────────────────────────────────────────────────────

class TestSelectedRows:
    def test_returns_only_checked(self):
        assert _selected_rows({0: True, 1: False, 2: True}) == [0, 2]

    def test_empty(self):
        assert _selected_rows({}) == []

    def test_none_checked(self):
        assert _selected_rows({0: False, 1: False}) == []

    def test_preserves_dict_order(self):
        assert _selected_rows({5: True, 1: True, 3: True}) == [5, 1, 3]


# ─────────────────────────────────────────────────────────────────────
# _selected_db_values
# ─────────────────────────────────────────────────────────────────────

class TestSelectedDbValues:
    def test_collects_valid_values(self, standard_table):
        assert _selected_db_values(standard_table, [0, 1, 2], 1) == [80.0, 60.0, 100.0]

    def test_only_selected_rows(self, standard_table):
        assert _selected_db_values(standard_table, [1], 1) == [60.0]

    @pytest.mark.parametrize("bad", ["—", "", "abc", "nan", "inf", "-inf"])
    def test_skips_invalid_and_non_finite(self, qapp, bad):
        table = make_table([DB_HEADER], [["50"], [bad], ["70"]])
        assert _selected_db_values(table, [0, 1, 2], 0) == [50.0, 70.0]

    def test_skips_empty_cells(self, qapp):
        table = make_table([DB_HEADER], [["50"], [None]])
        assert _selected_db_values(table, [0, 1], 0) == [50.0]

    def test_accepts_negative_and_decimal_values(self, qapp):
        table = make_table([DB_HEADER], [["-10.5"], [" 42 "]])
        assert _selected_db_values(table, [0, 1], 0) == [-10.5, 42.0]


# ─────────────────────────────────────────────────────────────────────
# _collect_io_data
# ─────────────────────────────────────────────────────────────────────

class TestCollectIoData:
    def collect(self, table, rows, ptp_col=2, db_min=-1e6, db_max=1e6):
        return _collect_io_data(
            table=table,
            selected_rows=rows,
            db_col=1,
            ptp_col=ptp_col,
            db_min=db_min,
            db_max=db_max,
        )

    def test_sorted_by_db(self, standard_table):
        x, y = self.collect(standard_table, [0, 1, 2, 3])
        np.testing.assert_array_equal(x, [60, 70, 80, 100])
        np.testing.assert_array_equal(y, [80.0, 95.0, 120.5, 200.0])

    def test_returns_float_arrays(self, standard_table):
        x, y = self.collect(standard_table, [0, 1])
        assert isinstance(x, np.ndarray) and x.dtype == float
        assert isinstance(y, np.ndarray) and y.dtype == float

    def test_skips_placeholder_ptp(self, standard_table):
        # Row 3 has "—" in Vestibular PtP.
        x, y = self.collect(standard_table, [0, 1, 2, 3], ptp_col=3)
        np.testing.assert_array_equal(x, [60, 80, 100])
        np.testing.assert_array_equal(y, [20.0, 30.0, 45.0])

    def test_ignores_unselected_rows(self, standard_table):
        x, _ = self.collect(standard_table, [0, 2])
        np.testing.assert_array_equal(x, [80, 100])

    def test_db_range_is_inclusive(self, standard_table):
        x, _ = self.collect(standard_table, [0, 1, 2, 3], db_min=70, db_max=80)
        np.testing.assert_array_equal(x, [70, 80])

    def test_repeated_db_kept_separate_and_stable(self, qapp):
        table = make_table(
            STANDARD_HEADERS,
            [
                ["a", "80", "1.0", "0"],
                ["b", "60", "2.0", "0"],
                ["c", "80", "3.0", "0"],
                ["d", "80", "4.0", "0"],
            ],
        )
        x, y = self.collect(table, [0, 1, 2, 3])
        np.testing.assert_array_equal(x, [60, 80, 80, 80])
        # Stable sort: duplicates keep their original row order, no averaging.
        np.testing.assert_array_equal(y, [2.0, 1.0, 3.0, 4.0])

    @pytest.mark.parametrize("bad_db, bad_ptp", [("nan", "1"), ("1", "inf"), ("x", "1"), ("1", "")])
    def test_skips_non_finite_or_unparseable(self, qapp, bad_db, bad_ptp):
        table = make_table(
            STANDARD_HEADERS,
            [["a", "60", "1", "0"], ["b", bad_db, bad_ptp, "0"], ["c", "70", "2", "0"]],
        )
        x, y = self.collect(table, [0, 1, 2])
        np.testing.assert_array_equal(x, [60, 70])
        np.testing.assert_array_equal(y, [1, 2])

    def test_skips_missing_items(self, qapp):
        table = make_table(
            STANDARD_HEADERS,
            [["a", "60", "1", "0"], ["b", None, "5", "0"], ["c", "70", None, "0"], ["d", "80", "3", "0"]],
        )
        x, _ = self.collect(table, [0, 1, 2, 3])
        np.testing.assert_array_equal(x, [60, 80])

    def test_negative_ptp_values_are_kept(self, qapp):
        table = make_table(STANDARD_HEADERS, [["a", "60", "-5", "0"], ["b", "70", "0", "0"]])
        _, y = self.collect(table, [0, 1])
        np.testing.assert_array_equal(y, [-5, 0])

    def test_exactly_two_points_is_enough(self, standard_table):
        x, _ = self.collect(standard_table, [0, 1])
        assert len(x) == 2

    @pytest.mark.parametrize("rows", [[], [0]])
    def test_fewer_than_two_points_raises(self, standard_table, rows):
        with pytest.raises(ValueError, match="At least two"):
            self.collect(standard_table, rows)

    def test_range_excluding_all_but_one_raises(self, standard_table):
        with pytest.raises(ValueError):
            self.collect(standard_table, [0, 1, 2, 3], db_min=100, db_max=200)


# ─────────────────────────────────────────────────────────────────────
# IOCurveDialog
# ─────────────────────────────────────────────────────────────────────

class TestIOCurveDialog:
    def test_defaults_from_db_values(self, parent):
        dialog = IOCurveDialog(parent, [80.0, 60.0, 100.0])
        assert dialog.db_min.value() == 60.0
        assert dialog.db_max.value() == 100.0
        assert dialog.auto_y.isChecked()
        assert not dialog.y_min.isEnabled()
        assert not dialog.y_max.isEnabled()

    def test_measurement_options(self, parent):
        dialog = IOCurveDialog(parent, [60.0])
        options = [dialog.measurement_combo.itemText(i) for i in range(dialog.measurement_combo.count())]
        assert options == [COCHLEAR_HEADER, VESTIBULAR_HEADER, BOTH_HEADER]
        assert dialog.measurement_combo.currentText() == COCHLEAR_HEADER

    def test_unchecking_auto_y_enables_manual_fields(self, parent):
        dialog = IOCurveDialog(parent, [60.0])
        dialog.auto_y.setChecked(False)
        assert dialog.y_min.isEnabled()
        assert dialog.y_max.isEnabled()
        dialog.auto_y.setChecked(True)
        assert not dialog.y_min.isEnabled()
        assert not dialog.y_max.isEnabled()

    def test_settings_auto_y(self, parent):
        dialog = IOCurveDialog(parent, [60.0, 90.0])
        s = dialog.settings()
        assert s == IOCurveSettings(
            measurement_name=COCHLEAR_HEADER,
            db_min=60.0,
            db_max=90.0,
            auto_y=True,
            y_min=None,
            y_max=None,
        )

    def test_settings_manual_y(self, parent):
        dialog = IOCurveDialog(parent, [60.0, 90.0])
        dialog.measurement_combo.setCurrentText(BOTH_HEADER)
        dialog.db_min.setValue(65.0)
        dialog.db_max.setValue(85.0)
        dialog.auto_y.setChecked(False)
        dialog.y_min.setValue(10.0)
        dialog.y_max.setValue(250.0)
        s = dialog.settings()
        assert s.measurement_name == BOTH_HEADER
        assert (s.db_min, s.db_max) == (65.0, 85.0)
        assert s.auto_y is False
        assert (s.y_min, s.y_max) == (10.0, 250.0)

    def test_db_values_rounded_to_three_decimals(self, parent):
        dialog = IOCurveDialog(parent, [60.12345, 70.0])
        assert dialog.db_min.value() == pytest.approx(60.123)

    def test_empty_db_values_raises(self, parent):
        # Documents current behaviour: the caller must guard against this.
        with pytest.raises(ValueError):
            IOCurveDialog(parent, [])


# ─────────────────────────────────────────────────────────────────────
# _build_plot
# ─────────────────────────────────────────────────────────────────────

class TestBuildPlot:
    def build(self, settings=None, **kwargs):
        params = dict(
            x1=np.array([60.0, 80.0]),
            y1=np.array([1.0, 2.0]),
            label1=COCHLEAR_HEADER,
            color1=COCH_COLOR,
            settings=settings or make_settings(),
            panel_color=PANEL,
            border_color=BORDER,
            text_color=TEXT,
        )
        params.update(kwargs)
        return _build_plot(**params)

    def test_single_series(self, qapp):
        plot = self.build()
        items = data_items(plot)
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].xData, [60, 80])
        np.testing.assert_array_equal(items[0].yData, [1, 2])
        assert items[0].name() == COCHLEAR_HEADER

    def test_single_series_has_no_legend(self, qapp):
        plot = self.build()
        assert plot.plotItem.legend is None

    def test_single_title_and_labels(self, qapp):
        plot = self.build(settings=make_settings(measurement_name=VESTIBULAR_HEADER))
        assert VESTIBULAR_HEADER in plot.plotItem.titleLabel.text
        assert plot.getAxis("bottom").labelText == "dB SPL"
        assert plot.getAxis("left").labelText == VESTIBULAR_HEADER

    def test_two_series_with_legend(self, qapp):
        plot = self.build(
            settings=make_settings(measurement_name=BOTH_HEADER),
            x2=np.array([60.0, 90.0]),
            y2=np.array([5.0, 6.0]),
            label2=VESTIBULAR_HEADER,
            color2=VEST_COLOR,
        )
        items = data_items(plot)
        assert [i.name() for i in items] == [COCHLEAR_HEADER, VESTIBULAR_HEADER]
        assert plot.plotItem.legend is not None
        assert "Cochlear and Vestibular" in plot.plotItem.titleLabel.text
        assert plot.getAxis("left").labelText == "PtP"

    def test_second_series_skipped_without_label_or_color(self, qapp):
        plot = self.build(x2=np.array([1.0, 2.0]), y2=np.array([1.0, 2.0]))
        assert len(data_items(plot)) == 1

    def test_series_colours(self, qapp):
        plot = self.build(
            x2=np.array([1.0, 2.0]),
            y2=np.array([1.0, 2.0]),
            label2=VESTIBULAR_HEADER,
            color2=VEST_COLOR,
        )
        first, second = data_items(plot)
        assert first.opts["pen"].color().name() == COCH_COLOR
        assert second.opts["pen"].color().name() == VEST_COLOR

    def test_manual_y_range(self, qapp):
        plot = self.build(settings=make_settings(auto_y=False, y_min=-10.0, y_max=300.0))
        y_low, y_high = plot.viewRange()[1]
        assert y_low == pytest.approx(-10.0)
        assert y_high == pytest.approx(300.0)

    def test_auto_y_leaves_autorange_enabled(self, qapp):
        plot = self.build()
        assert plot.plotItem.vb.autoRangeEnabled()[1]

    def test_attach_hover_called_with_plot(self, qapp):
        hover = Mock()
        plot = self.build(attach_hover=hover)
        hover.assert_called_once_with(plot)


# ─────────────────────────────────────────────────────────────────────
# export_io_graph
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def svg_export_works(qapp) -> bool:
    """
    Some pyqtgraph/Qt combinations (e.g. pyqtgraph 0.14.0 with Qt 6.11) crash
    inside pyqtgraph's own SVGExporter on *any* plot. That is a library issue,
    not a bug in input_output.py, so real SVG-writing tests are skipped there.
    """
    import tempfile

    import pyqtgraph as pg
    import pyqtgraph.exporters

    probe = pg.PlotWidget()
    probe.plot([0, 1], [0, 1])
    probe.resize(200, 200)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pyqtgraph.exporters.SVGExporter(probe.plotItem).export(os.path.join(tmp, "p.svg"))
            return True
        except Exception:
            return False


@pytest.fixture
def requires_svg(svg_export_works):
    if not svg_export_works:
        pytest.skip("pyqtgraph's SVGExporter is broken with this Qt version (library issue)")


class TestExportIoGraph:
    @pytest.fixture
    def plot(self, qapp):
        plot = _build_plot(
            x1=np.array([60.0, 80.0, 100.0]),
            y1=np.array([1.0, 2.0, 3.0]),
            label1=COCHLEAR_HEADER,
            color1=COCH_COLOR,
            settings=make_settings(),
            panel_color=PANEL,
            border_color=BORDER,
            text_color=TEXT,
        )
        plot.resize(600, 400)
        return plot

    @pytest.mark.parametrize("name", ["graph.svg", "GRAPH.SVG"])
    def test_svg_dispatches_to_svg_exporter(self, plot, monkeypatch, name):
        svg_cls, img_cls = Mock(), Mock()
        monkeypatch.setattr(io_mod.pyqtgraph.exporters, "SVGExporter", svg_cls)
        monkeypatch.setattr(io_mod.pyqtgraph.exporters, "ImageExporter", img_cls)

        export_io_graph(plot, name)

        svg_cls.assert_called_once_with(plot.plotItem)
        svg_cls.return_value.export.assert_called_once_with(name)
        img_cls.assert_not_called()

    def test_png_dispatches_to_image_exporter_at_2400px(self, plot, monkeypatch):
        params = {}
        img_cls = Mock()
        img_cls.return_value.parameters.return_value = params
        svg_cls = Mock()
        monkeypatch.setattr(io_mod.pyqtgraph.exporters, "SVGExporter", svg_cls)
        monkeypatch.setattr(io_mod.pyqtgraph.exporters, "ImageExporter", img_cls)

        export_io_graph(plot, "graph.PNG")

        img_cls.assert_called_once_with(plot.plotItem)
        assert params["width"] == 2400
        img_cls.return_value.export.assert_called_once_with("graph.PNG")
        svg_cls.assert_not_called()

    @pytest.mark.usefixtures("requires_svg")
    def test_svg_export(self, plot, tmp_path):
        path = tmp_path / "graph.svg"
        export_io_graph(plot, str(path))
        content = path.read_text()
        assert "<svg" in content

    def test_png_export(self, plot, tmp_path):
        path = tmp_path / "graph.png"
        export_io_graph(plot, str(path))
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG")

    def test_png_export_width_is_2400(self, plot, tmp_path):
        from PyQt6.QtGui import QImage

        path = tmp_path / "graph.png"
        export_io_graph(plot, str(path))
        assert QImage(str(path)).width() == 2400

    def test_png_extension_is_case_insensitive(self, plot, tmp_path):
        path = tmp_path / "Graph.Png"
        export_io_graph(plot, str(path))
        assert path.read_bytes().startswith(b"\x89PNG")

    @pytest.mark.parametrize("name", ["graph.jpg", "graph.pdf", "graph", "graph.svg.txt"])
    def test_unsupported_extension_raises(self, plot, tmp_path, name):
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_io_graph(plot, str(tmp_path / name))
        assert not (tmp_path / name).exists()


# ─────────────────────────────────────────────────────────────────────
# build_io_curve_from_table — validation and early exits
# ─────────────────────────────────────────────────────────────────────

class TestWorkflowValidation:
    def test_no_rows_selected(self, parent, standard_table, messages, fake_dialog):
        fake_dialog()
        result = run_workflow(parent, standard_table, {0: False, 1: False})
        assert result is None
        assert messages.calls[0][0] == "information"
        assert "Select the recordings" in messages.calls[0][2]

    def test_missing_db_column(self, parent, qapp, messages, fake_dialog):
        fake_dialog()
        table = make_table(["File", COCHLEAR_HEADER], [["a", "1"], ["b", "2"]])
        assert run_workflow(parent, table, all_checked(2)) is None
        assert messages.calls == [("warning", "I/O Graph", f"Could not find the '{DB_HEADER}' column.")]

    def test_no_valid_db_values(self, parent, qapp, messages, fake_dialog):
        fake_dialog()
        table = make_table(STANDARD_HEADERS, [["a", "—", "1", "1"], ["b", "", "2", "2"]])
        assert run_workflow(parent, table, all_checked(2)) is None
        assert "no valid dB values" in messages.calls[0][2]

    def test_dialog_receives_selected_db_values(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(accepted=False)
        run_workflow(parent, standard_table, {0: True, 1: False, 2: True, 3: False})
        assert fake_dialog.received_db_values == [80.0, 100.0]

    def test_cancelled_dialog(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(accepted=False)
        assert run_workflow(parent, standard_table, all_checked(4)) is None
        assert messages.calls == []

    def test_db_min_greater_than_max(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(db_min=90, db_max=80))
        assert run_workflow(parent, standard_table, all_checked(4)) is None
        assert "Minimum dB" in messages.calls[0][2]

    def test_db_min_equal_max_is_allowed_but_needs_two_points(self, parent, standard_table, messages, fake_dialog):
        # Passes range validation, then fails because only one point lies at 80 dB.
        fake_dialog(make_settings(db_min=80, db_max=80))
        assert run_workflow(parent, standard_table, all_checked(4)) is None
        assert "At least two" in messages.calls[0][2]

    @pytest.mark.parametrize("y_min, y_max", [(100.0, 50.0), (50.0, 50.0)])
    def test_invalid_manual_y_range(self, parent, standard_table, messages, fake_dialog, y_min, y_max):
        fake_dialog(make_settings(auto_y=False, y_min=y_min, y_max=y_max))
        assert run_workflow(parent, standard_table, all_checked(4)) is None
        assert "Y-axis minimum" in messages.calls[0][2]

    def test_missing_ptp_column_single_mode(self, parent, qapp, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=VESTIBULAR_HEADER))
        table = make_table(["File", DB_HEADER, COCHLEAR_HEADER], [["a", "60", "1"], ["b", "70", "2"]])
        assert run_workflow(parent, table, all_checked(2)) is None
        assert f"'{VESTIBULAR_HEADER}'" in messages.calls[0][2]

    def test_too_few_points_single_mode(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=COCHLEAR_HEADER))
        assert run_workflow(parent, standard_table, {0: True}) is None
        assert "At least two" in messages.calls[0][2]

    def test_missing_column_both_mode(self, parent, qapp, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        table = make_table(["File", DB_HEADER, COCHLEAR_HEADER], [["a", "60", "1"], ["b", "70", "2"]])
        assert run_workflow(parent, table, all_checked(2)) is None
        assert "Could not find both" in messages.calls[0][2]

    def test_no_plottable_data_both_mode(self, parent, qapp, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        table = make_table(STANDARD_HEADERS, [["a", "60", "—", "—"], ["b", "70", "—", "—"]])
        assert run_workflow(parent, table, all_checked(2)) is None
        assert "No valid Cochlear or Vestibular" in messages.calls[0][2]

    def test_hover_not_called_on_failure(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(accepted=False)
        hover = Mock()
        run_workflow(parent, standard_table, all_checked(4), attach_hover=hover)
        hover.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# build_io_curve_from_table — successful runs
# ─────────────────────────────────────────────────────────────────────

class TestWorkflowSuccess:
    def test_single_cochlear(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=COCHLEAR_HEADER))
        result = run_workflow(parent, standard_table, all_checked(4))

        assert result is not None
        assert messages.calls == []
        assert result.title == f"I/O Graph — {COCHLEAR_HEADER}"
        assert result.measurement_name == COCHLEAR_HEADER
        assert result.point_count == 4

        (item,) = data_items(result.plot)
        np.testing.assert_array_equal(item.xData, [60, 70, 80, 100])
        np.testing.assert_array_equal(item.yData, [80.0, 95.0, 120.5, 200.0])
        assert item.opts["pen"].color().name() == COCH_COLOR

    def test_single_vestibular_uses_vestibular_colour(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=VESTIBULAR_HEADER))
        result = run_workflow(parent, standard_table, all_checked(4))

        assert result.point_count == 3  # row 3 has a "—" placeholder
        (item,) = data_items(result.plot)
        assert item.opts["pen"].color().name() == VEST_COLOR

    def test_db_range_filter_applied(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(db_min=65, db_max=85))
        result = run_workflow(parent, standard_table, all_checked(4))
        (item,) = data_items(result.plot)
        np.testing.assert_array_equal(item.xData, [70, 80])

    def test_only_checked_rows_used(self, parent, standard_table, messages, fake_dialog):
        fake_dialog()
        result = run_workflow(parent, standard_table, {0: True, 1: False, 2: True, 3: False})
        (item,) = data_items(result.plot)
        np.testing.assert_array_equal(item.xData, [80, 100])

    def test_manual_y_range_applied(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(auto_y=False, y_min=0.0, y_max=500.0))
        result = run_workflow(parent, standard_table, all_checked(4))
        assert result.plot.viewRange()[1] == pytest.approx([0.0, 500.0])

    def test_both_series(self, parent, standard_table, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        result = run_workflow(parent, standard_table, all_checked(4))

        assert result.title == "I/O Graph — Cochlear and Vestibular PtP"
        assert result.measurement_name == BOTH_HEADER
        assert result.point_count == 4 + 3

        coch, vest = data_items(result.plot)
        assert coch.name() == COCHLEAR_HEADER
        assert vest.name() == VESTIBULAR_HEADER
        np.testing.assert_array_equal(vest.xData, [60, 80, 100])
        assert result.plot.plotItem.legend is not None

    def test_both_mode_with_only_cochlear_data(self, parent, qapp, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        table = make_table(STANDARD_HEADERS, [["a", "60", "1", "—"], ["b", "70", "2", "—"]])
        result = run_workflow(parent, table, all_checked(2))

        assert result.point_count == 2
        (item,) = data_items(result.plot)
        assert item.name() == COCHLEAR_HEADER
        assert result.plot.plotItem.legend is None

    def test_both_mode_with_only_vestibular_data(self, parent, qapp, messages, fake_dialog):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        table = make_table(STANDARD_HEADERS, [["a", "60", "—", "5"], ["b", "70", "—", "6"]])
        result = run_workflow(parent, table, all_checked(2))

        assert result.point_count == 2
        items = data_items(result.plot)
        # Current behaviour: an empty Cochlear series is still added.
        assert [i.name() for i in items] == [COCHLEAR_HEADER, VESTIBULAR_HEADER]
        assert len(items[0].xData if items[0].xData is not None else []) == 0
        np.testing.assert_array_equal(items[1].yData, [5, 6])

    def test_both_mode_drops_series_with_single_point(self, parent, qapp, messages, fake_dialog):
        # Current behaviour: a series with just one valid point is silently omitted.
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        table = make_table(
            STANDARD_HEADERS,
            [["a", "60", "1", "5"], ["b", "70", "2", "—"], ["c", "80", "3", "—"]],
        )
        result = run_workflow(parent, table, all_checked(3))
        assert result.point_count == 3
        assert len(data_items(result.plot)) == 1
        assert messages.calls == []

    def test_attach_hover_called_once(self, parent, standard_table, messages, fake_dialog):
        fake_dialog()
        hover = Mock()
        result = run_workflow(parent, standard_table, all_checked(4), attach_hover=hover)
        hover.assert_called_once_with(result.plot)

    def test_result_plot_is_exportable_as_png(self, parent, standard_table, messages, fake_dialog, tmp_path):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        result = run_workflow(parent, standard_table, all_checked(4))
        result.plot.resize(600, 400)
        path = tmp_path / "out.png"
        export_io_graph(result.plot, str(path))
        assert path.stat().st_size > 0

    @pytest.mark.usefixtures("requires_svg")
    def test_result_plot_is_exportable_as_svg(self, parent, standard_table, messages, fake_dialog, tmp_path):
        fake_dialog(make_settings(measurement_name=BOTH_HEADER))
        result = run_workflow(parent, standard_table, all_checked(4))
        result.plot.resize(600, 400)
        path = tmp_path / "out.svg"
        export_io_graph(result.plot, str(path))
        assert "<svg" in path.read_text()
