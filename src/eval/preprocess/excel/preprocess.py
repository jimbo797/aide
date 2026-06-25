import io
from pathlib import Path
import json
import pandas as pd

import msoffcrypto
import openpyxl
from openpyxl.worksheet.worksheet import Worksheet


def get_workbook(filepath: str, password: str | None = None, data_only: bool = True) -> openpyxl.Workbook:
    content = get_excel_bytes(filepath, password=password)
    return openpyxl.load_workbook(
        io.BytesIO(content),
        data_only=data_only,
        keep_vba=filepath.lower().endswith(".xlsm"),
    )


def get_excel_bytes(filename: str, password: str | None = None) -> bytes:
    with open(filename, "rb") as f:
        content = f.read()

    office_file = msoffcrypto.OfficeFile(io.BytesIO(content))
    if not office_file.is_encrypted():
        return content

    if not password:
        raise ValueError(f"Password required for encrypted file: {filename}")

    office_file.load_key(password=password)
    decrypted = io.BytesIO()
    office_file.decrypt(decrypted)
    return decrypted.getvalue()


def get_excel_formula_cells(ws: Worksheet) -> list[tuple[str, str]]:
    formula_cells = []
    for coord, value in (
        (cell.coordinate, cell.value)
        for row in ws.iter_rows()
        for cell in row
        if isinstance(cell.value, str) and cell.value.startswith("=")
    ):
        formula_cells.append((coord, value))
    return formula_cells


def get_series_label(series) -> str | None:
    title = series.title
    if title is None:
        return None
    if title.v is not None:
        return str(title.v)
    str_ref = title.strRef
    if str_ref and str_ref.strCache and str_ref.strCache.pt:
        return str_ref.strCache.pt[0].v
    return None


def get_excel_charts(ws: Worksheet) -> list[dict]:
    charts = []
    for chart in ws._charts:
        chart_data = {
            "type": type(chart).__name__,
            "series": [],
        }
        for series in chart.series:
            values: list = []
            values_ref = None
            if series.val and series.val.numRef:
                values_ref = series.val.numRef.f
                if series.val.numRef.numCache:
                    values = [p.v for p in series.val.numRef.numCache.pt]
            chart_data["series"].append(
                {
                    "label": get_series_label(series),
                    "label_ref": series.title.strRef.f if series.title and series.title.strRef else None,
                    "values": values,
                    "values_ref": values_ref,
                }
            )
        charts.append(chart_data)
    return charts


# preprocesses a downloaded excel sheet
def preprocess_excel_sheet(filepath: str, alias: str, preprocess_dir: Path, sheet_name: str | None = None, password: str | None = None) -> None:
    content = get_excel_bytes(filepath, password=password)
    xl = pd.ExcelFile(io.BytesIO(content))

    wb_data = get_workbook(filepath, password=password, data_only=True)
    wb_formulas = get_workbook(filepath, password=password, data_only=False)

    if sheet_name is None:
        sheet_name = xl.sheet_names[0]
    
    df = xl.parse(sheet_name)
    formula_cells = get_excel_formula_cells(wb_formulas[sheet_name])
    charts = get_excel_charts(wb_data[sheet_name])

    out_dir = preprocess_dir / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "excel.json", "w") as f:
        excel_data = {
            "sheet_name": sheet_name,
            "formula_cells": formula_cells,
            "sheet_values_csv": df.to_csv(index=False),
            "charts": charts,
        }
        json.dump(excel_data, f, indent=2)


if __name__ == "__main__":
    preprocess_excel_sheet(
        filepath="gsu-materials/gsu-grading-spring-2/Video_Data/Kamya Kelly Video Data.xlsm",
        alias="kamya_kelly",
        preprocess_dir=Path("aide/out/preprocess"),
        sheet_name="Forecast",
    )
