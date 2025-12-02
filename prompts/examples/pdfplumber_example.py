import pandas as pd

EXPECTED_COLS = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]


def parse(pdf_path: str) -> pd.DataFrame:
    try:
        import pdfplumber  # type: ignore
    except Exception as e:
        raise ImportError("pdfplumber not installed or failed to import") from e

    # Extract all tables from all pages
    rows: list[list[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    # Normalize cells to stripped strings; keep empty cells as ""
                    rows.append([("" if c is None else str(c).strip()) for c in row])

    if not rows:
        raise ValueError("No tables extracted via pdfplumber; need tuned parsing")

    raw = pd.DataFrame(rows)

    # Treat empty strings as missing for dropping fully empty rows/cols
    raw = raw.replace(r"^\s*$", pd.NA, regex=True)
    raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
    # Re-strip after NA cleanup (keeps symmetry with camelot example)
    raw = raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Use first row as header if all string-like; then remove repeated header rows
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

    # Map header aliases to expected columns
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

    # Ensure exact columns and order
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[EXPECTED_COLS]

    # Normalize types
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