def add_columns(expected_cols: str, template: str) -> str:
    lines = template.splitlines()
    insert_pos = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            insert_pos = i + 1
    # Avoid duplicate blank lines
    expected_line = expected_cols.rstrip("\n")
    new_lines = lines[:insert_pos] + [expected_line, ""] + lines[insert_pos:]
    return "\n".join(new_lines)

add_columns('EXPECTED_COLS = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]', '''# Compare this snippet from custom_parsers/icici_parser.py:''')