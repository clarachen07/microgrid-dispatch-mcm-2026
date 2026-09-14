"""Export the five result workbooks required by the modelling problem."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from .data import DEFAULT_DATA_DIR, load_data

ROOT = Path(__file__).resolve().parents[2]
CHINESE = {"q2": "问题二", "q3": "问题三", "q4_2": "波动电价下的问题二", "q4_3": "波动电价下的问题三"}
OUTPUT_NAMES = {"q1": "result1.xlsx", "q2": "result2.xlsx", "q3": "result3.xlsx", "q4_2": "result4-2.xlsx", "q4_3": "result4-3.xlsx"}


def clock(slot: int) -> str:
    return f"{slot // 6:02d}:{slot % 6 * 10:02d}"


def period(start: int, end: int) -> str:
    return f"{clock(start)}-{clock(end)}"


def events(values: np.ndarray, eps: float = 1e-7) -> list[tuple[int, int, float]]:
    on = np.asarray(values) > eps
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    ends = np.flatnonzero(on & ~np.r_[on[1:], False]) + 1
    return [(int(start), int(end), float(values[start:end].sum())) for start, end in zip(starts, ends)]


def fetch(name: str, folder: str = "artifacts/runs") -> tuple[dict[str, np.ndarray], dict]:
    base = ROOT / folder / name
    with np.load(base / "ledger.npz") as archive:
        log = {key: archive[key].copy() for key in archive.files}
    return log, json.loads((base / "summary.json").read_text(encoding="utf-8"))


def numeric_style(sheet) -> None:
    sheet.freeze_panes = "B2" if sheet.max_column > 20 else "A2"
    sheet.row_dimensions[1].height = 30
    for cell in sheet[1]:
        cell.font = Font(name="Arial", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="right" if isinstance(cell.value, (float, int)) else "left", vertical="center")
            if isinstance(cell.value, (float, int)):
                cell.number_format = "0.0000"
    sheet.column_dimensions["A"].width = 14


def export_workbook(name: str, log: dict[str, np.ndarray], out: Path) -> None:
    book = load_workbook(DEFAULT_DATA_DIR / "附件5" / OUTPUT_NAMES[name])
    book.properties.creator = ""
    book.properties.lastModifiedBy = ""
    book.properties.title = CHINESE.get(name, "问题一") + "计算结果"
    plan = book["计划购电量"]
    if name == "q1":
        for slot in range(144):
            plan.cell(slot + 2, 1, period(slot, slot + 1))
            plan.cell(slot + 2, 2, float(log["plan"][slot]))
        plan.column_dimensions["A"].width = 22
        plan.column_dimensions["B"].width = 20
        storage = book["充放电量"]
        for block in range(6):
            start, end = block * 24, (block + 1) * 24
            storage.cell(block + 2, 1, period(start, end))
            storage.cell(block + 2, 2, float(log["charge"][start:end].sum()))
            storage.cell(block + 2, 3, float(log["discharge"][start:end].sum()))
        storage.cell(2, 4, "00:00")
        storage.cell(2, 5, float(log["soc"][0]))
        storage.cell(3, 4, "24:00")
        storage.cell(3, 5, float(log["soc"][-1]))
    else:
        for label, key in (("计划购电量", "plan"), ("调整购电量", "effective")):
            if label not in book.sheetnames:
                continue
            sheet = book[label]
            for slot in range(144):
                sheet.cell(1, slot + 2, period(slot, slot + 1))
            for day in range(31, 365):
                row, start, end = day - 29, day * 144, (day + 1) * 144
                sheet.cell(row, 1, dt.datetime(2025, 1, 1) + dt.timedelta(days=day))
                sheet.cell(row, 1).number_format = "yyyy-mm-dd"
                for slot, value in enumerate(log[key][start:end]):
                    sheet.cell(row, slot + 2, float(value))
                sheet.cell(row, 146, float(log[key][start:end].sum()))
                cost = log["base_fee"][start:end].sum()
                if label == "调整购电量":
                    cost += log["up_fee"][start:end].sum() + log["down_fee"][start:end].sum()
                sheet.cell(row, 147, float(cost))
            for column in range(2, 146):
                sheet.column_dimensions[get_column_letter(column)].width = 16
            sheet.column_dimensions["EP"].width = 20
            sheet.column_dimensions["EQ"].width = 20

        storage = book["充放电量"]
        storage.delete_rows(2, storage.max_row)
        for day in range(31, 365):
            date = dt.datetime(2025, 1, 1) + dt.timedelta(days=day)
            for block in range(6):
                start, end = day * 144 + block * 24, day * 144 + (block + 1) * 24
                storage.append([date if block == 0 else None, period(block * 24, (block + 1) * 24),
                                float(log["charge"][start:end].sum()), float(log["discharge"][start:end].sum()),
                                "00:00" if block == 0 else "24:00" if block == 1 else None,
                                float(log["soc_start"][day * 144]) if block == 0 else float(log["soc_end"][(day + 1) * 144 - 1]) if block == 1 else None])
                if block == 0:
                    storage.cell(storage.max_row, 1).number_format = "yyyy-mm-dd"

        urgent = book["紧急购电量"]
        urgent.delete_rows(2, urgent.max_row)
        for day in range(31, 365):
            date = dt.datetime(2025, 1, 1) + dt.timedelta(days=day)
            entries = events(log["emergency"][day * 144:(day + 1) * 144]) or [(None, None, 0.0)]
            for index, (start, end, value) in enumerate(entries):
                urgent.append([date if index == 0 else None, "无" if start is None else period(start, end), value])
                if index == 0:
                    urgent.cell(urgent.max_row, 1).number_format = "yyyy-mm-dd"
        urgent.column_dimensions["B"].width = 23
        urgent.column_dimensions["C"].width = 23

    for sheet in book:
        numeric_style(sheet)
        if sheet.max_column <= 6:
            for column in range(2, sheet.max_column + 1):
                sheet.column_dimensions[get_column_letter(column)].width = 20
    book.save(out / OUTPUT_NAMES[name])


def export_all() -> None:
    """Write the five official result workbooks from completed primary runs."""
    load_data()  # Fail early when the official inputs are unavailable or invalid.
    out = ROOT / "artifacts" / "exports"
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        log, _ = fetch(name)
        export_workbook(name, log, out)
    print("Exported the five required result workbooks.", flush=True)
