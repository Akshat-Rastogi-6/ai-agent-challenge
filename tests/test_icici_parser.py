from pathlib import Path
import importlib
import pandas as pd
import pandas.testing as pdt
import pytest

DATA_DIR = Path("data") / "icici"
PDF_PATH = DATA_DIR / "icici sample.pdf"
CSV_DIR = DATA_DIR / "result.csv"

def test_icici_parser_strict():
    mod = importlib.import_module("custom_parsers.icici_parser")
    assert hasattr(mod, "parse"), "custom_parsers/icici_parser.py must define parse(pdf_path) -> DataFrame"

    actual_df = mod.parse(str(PDF_PATH))

    expected_df = pd.read_csv(CSV_DIR)

    assert list(actual_df.columns) == list(expected_df.columns), (
        f"Columns differ. \n Exepcted: {list(expected_df.columns)}\nActual:   {list(actual_df.columns)}"
    )

    assert actual_df.shape == expected_df.shape, (
        f"Shape  differs. \n Exepcted: {expected_df.shape}\nActual:   {actual_df.shape}"
    )

    if not actual_df.equals(expected_df):
        try:
            pdt.assert_frame_equal(actual_df, expected_df, check_dtype=True, check_like=False, obj="ICICI parse")
        except AssertionError as e:
            raise AssertionError(f"DataFrames differ:\n{e}") from e