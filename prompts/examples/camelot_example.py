# ...existing code...
import pandas as pd

EXPECTED_COLS = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]


def parse(pdf_path: str) -> pd.DataFrame:
    try:
        import camelot   #type: ignore
    except Exception as e:
        raise ImportError("Camelot not installed or failed to import") from e

    tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    if not tables:
        raise ValueError("No tables detected in PDF; try different flavor or pdfplumber fallback")

    frames = [t.df for t in tables if not t.df.empty]
    if not frames:
        raise ValueError("Extracted tables are empty")
    raw = pd.concat(frames, ignore_index=True)

    raw = raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # Use first row as header if all string-like; then remove any repeated header rows appearing later
    header_tokens: list[str] | None = None
    if raw.shape[0] > 1 and all(isinstance(x, str) and x.strip() for x in raw.iloc[0].tolist()):
        header_tokens = [str(c).strip() for c in raw.iloc[0].tolist()]
        raw.columns = header_tokens
        raw = raw.iloc[1:].reset_index(drop=True)

    if header_tokens:
        # Drop subsequent rows that exactly repeat the header (multi-page repeated headers)
        duplicate_header_mask = raw.apply(
            lambda r: [("" if v is None else str(v).strip()) for v in r.tolist()] == header_tokens,
            axis=1,
        )
        if duplicate_header_mask.any():
            raw = raw.loc[~duplicate_header_mask].reset_index(drop=True)

    aliases = {
        "Txn Date": "Date",
        "Transaction Date": "Date",
        "Narration": "Description",
        "Withdrawal Amount": "Debit Amt",
        "Withdrawal": "Debit Amt",
        "Debit": "Debit Amt",
        "Deposit Amount": "Credit Amt",
        "Deposit": "Credit Amt",
        "Credit": "Credit Amt",
        "Closing Balance": "Balance",
        "Balance Amount": "Balance",
    }

    rename_map: dict[str, str] = {}
    for c in raw.columns:
        tgt = aliases.get(c)
        if tgt in EXPECTED_COLS:
            rename_map[c] = tgt
    df = raw.rename(columns=rename_map)

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[EXPECTED_COLS]

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        except Exception:
            pass

    numeric_cols = [c for c in ["Debit Amt", "Credit Amt", "Amount", "Balance"] if c in df.columns]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("INR", "", regex=False)
            .str.replace("Dr", "", regex=False)
            .str.replace("Cr", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df