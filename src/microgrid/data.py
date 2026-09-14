"""Read the supplied workbooks without changing their values or templates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "official-input"
LEGACY_DATA_DIR = PROJECT_ROOT / "C题" / "附件"


def canonical_intervals() -> list[str]:
    """The model uses [00:00,00:10), ..., [23:50,24:00)."""
    def clock(minutes: int) -> str:
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    return [f"{clock(10*i)}-{clock(10*(i+1))}" for i in range(144)]


def parse_time_minutes(value: Any) -> int:
    if isinstance(value, (datetime, time)):
        return value.hour * 60 + value.minute
    text = str(value).strip()
    carry = 1440 if "+1" in text else 0
    text = text.replace("+1", "")
    hour, minute = text.split(":")[:2]
    return int(hour) * 60 + int(minute) + carry


@dataclass
class DataBundle:
    dates: np.ndarray
    load: np.ndarray
    pv: np.ndarray
    price: np.ndarray
    prior_load: np.ndarray
    prior_pv: np.ndarray
    prior_price: np.ndarray
    forecasts: np.ndarray
    source_dir: Path | None = None

    @property
    def flat_load(self) -> np.ndarray:
        return self.load.reshape(-1)

    @property
    def flat_pv(self) -> np.ndarray:
        return self.pv.reshape(-1)

    @property
    def flat_price(self) -> np.ndarray:
        return self.price.reshape(-1)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_steps(self) -> int:
        return self.n_days * 144

    @property
    def start_date(self) -> np.datetime64:
        return self.dates[0].astype("datetime64[D]")

    def audit(self) -> dict[str, Any]:
        return {
            "days": self.n_days,
            "start": str(self.dates[0]), "end": str(self.dates[-1]),
            "steps_per_day": 144, "step_hours": 1 / 6,
            "input_timestamp_convention": "right endpoint of preceding 10-minute interval",
            "output_interval_convention": "00:00-00:10 through 23:50-24:00",
            "template_offset_minutes_corrected": 10,
            "load_min_kw": float(self.load.min()), "load_max_kw": float(self.load.max()),
            "pv_min_kw": float(self.pv.min()), "pv_max_kw": float(self.pv.max()),
            "price_min": float(self.price.min()), "price_max": float(self.price.max()),
            "load_energy_kwh": float(self.load.sum() / 6),
            "pv_energy_kwh": float(self.pv.sum() / 6),
            "output_days": int(np.sum(self.dates >= np.datetime64("2025-02-01"))),
        }


def _rows(path: Path, sheet: str | None = None) -> list[tuple]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list((book[sheet] if sheet else book.active).values)
    finally:
        book.close()


def _resolve_data_dir(path: str | Path | None = None) -> Path:
    """Resolve project data first, with legacy support-package placement as fallback."""
    candidates = [Path(path)] if path is not None else [DEFAULT_DATA_DIR, LEGACY_DATA_DIR]
    for candidate in candidates:
        if (candidate / "附件1.xlsx").is_file():
            return candidate
        nested = candidate / "C题" / "附件"
        if (nested / "附件1.xlsx").is_file():
            return nested
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"官方附件未找到；已检查：{searched}")


def load_data(path: str | Path | None = None) -> DataBundle:
    path = _resolve_data_dir(path)
    prior_rows = _rows(path / "附件1.xlsx")
    times = [parse_time_minutes(row[0]) for row in prior_rows[1:]]
    if times != list(range(10, 1441, 10)):
        raise ValueError("附件1时间点不是00:10至24:00的144个连续时点")
    prior = np.asarray([row[1:4] for row in prior_rows[1:]], dtype=float)
    load_rows = _rows(path / "附件2.xlsx", "小区负载")
    pv_rows = _rows(path / "附件2.xlsx", "光伏发电实际功率")
    price_rows = _rows(path / "附件4.xlsx")
    dates = np.asarray([np.datetime64(row[0], "D") for row in load_rows[1:]])
    arrays = []
    for rows in (load_rows, pv_rows, price_rows):
        matrix_dates = np.asarray([np.datetime64(row[0], "D") for row in rows[1:]])
        if not np.array_equal(matrix_dates, dates):
            raise ValueError("附件2与附件4日期不一致")
        if [parse_time_minutes(v) for v in rows[0][1:]] != times:
            raise ValueError("输入工作表时间轴不一致")
        arrays.append(np.asarray([row[1:] for row in rows[1:]], dtype=float))
    if len(np.unique(dates)) != len(dates) or np.any(np.diff(dates).astype(int) != 1):
        raise ValueError("输入日期存在重复或缺失")
    predictions = np.full((len(dates), 4, 24), np.nan)
    date_map = {str(date): i for i, date in enumerate(dates)}
    issue_date = None
    for row in _rows(path / "附件3.xlsx")[1:]:
        if row[0]:
            issue_date = np.datetime64(datetime.strptime(str(row[0]), "%Y-%m-%d"), "D")
        if issue_date is None:
            raise ValueError("附件3首条预报缺少日期")
        minute = parse_time_minutes(row[1])
        if minute not in (0, 360, 720, 1080):
            raise ValueError("附件3存在非规定发布时刻")
        day = date_map[str(issue_date)]
        issue = minute // 360
        if np.isfinite(predictions[day, issue]).any():
            raise ValueError("附件3存在重复预报")
        predictions[day, issue] = row[2:26]
    for array in (*arrays, prior, predictions):
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("输入存在缺失、非有限或负数")
    if arrays[0].shape != (len(dates), 144) or np.any(arrays[2] <= 0):
        raise ValueError("输入形状错误或电价非正")
    return DataBundle(dates, arrays[0], arrays[1], arrays[2],
                      prior[:, 1], prior[:, 2], prior[:, 0], predictions, path)


def template_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Report the original template labels and their documented correction."""
    result: dict[str, Any] = {
        "model_intervals": canonical_intervals(),
        "original_purchase_intervals_shifted_minutes": 10,
        "correction": "Keep the 144 data positions; label their preceding intervals, beginning at 00:00.",
        "files": {},
    }
    template_dir = _resolve_data_dir(path) / "附件5" if path is None else Path(path)
    for file in sorted(template_dir.glob("*.xlsx")):
        book = openpyxl.load_workbook(file, data_only=True)
        result["files"][file.name] = {
            sheet.title: {
                "rows": sheet.max_row, "columns": sheet.max_column,
                "merged_cells": [str(v) for v in sheet.merged_cells.ranges],
                "header": [str(cell.value) if cell.value is not None else None for cell in sheet[1]],
            }
            for sheet in book
        }
        book.close()
    return result
