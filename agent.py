#!/usr/bin/env python3
# Agent loop with Gemini (LangChain) + LangGraph:
# plans → calls tools (code/file I/O, pytest) → observes results → refines itself

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, Optional
import importlib
import pandas as pd

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

load_dotenv(".env")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent
CUSTOM_DIR = PROJECT_ROOT / "custom_parsers"
EXAMPLE_DIR = PROJECT_ROOT / "prompts" / "examples"  # << few-shot examples directory

# Add small logging helpers
def _log(msg: str) -> None:
    print(f"[agent] {msg}", flush=True)

def _tail(text: str, lines: int = 40) -> str:
    txt = (text or "").strip().splitlines()
    return "\n".join(txt[-lines:])

# Keep attempts configurable
MAX_ATTEMPTS = int(os.getenv("AGENT_MAX_ATTEMPTS", "3"))

# ---------- Planning / Tools ----------

def plan(target: str) -> dict:
    data_dir = PROJECT_ROOT / "data" / target
    pdf = next(data_dir.glob("*.pdf"), None)
    csv_path = next(data_dir.glob("*.csv"), None)
    if not pdf:
        raise FileNotFoundError(f"Missing sample PDF in {data_dir}, found pdf={pdf}")
    return {
        "target": target,
        "pdf_path": str(pdf),
        "csv_path": str(csv_path) if csv_path else "",
        "parser_path": str(CUSTOM_DIR / f"{target}_parser.py"),
    }

def read_csv_header(csv_path: str) -> list[str]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])

def write_text(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def run_pytest(test_path: Optional[str]) -> tuple[int, str, str]:
    args = ["-m", "pytest", "-q"]
    if test_path:
        args.append(test_path)
    proc = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return proc.returncode, proc.stdout, proc.stderr

def run_builtin_eval(target: str, pdf_path: str, csv_path: str) -> tuple[int, str, str]:
    """
    Fallback evaluator for new banks: imports the generated parser and compares
    against the CSV in data/{target}/result.csv. Returns (exit, stdout, stderr).
    """
    try:
        mod_name = f"custom_parsers.{target}_parser"
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "parse"):
            return (1, "", f"Module {mod_name} missing parse(pdf_path) function.")
        actual_df = mod.parse(str(pdf_path))
        expected_df = pd.read_csv(csv_path)
        if "Date" in expected_df.columns:
            expected_df["Date"] = pd.to_datetime(expected_df["Date"], dayfirst=True, errors="coerce")

        # Columns check
        if list(actual_df.columns) != list(expected_df.columns):
            return (
                1,
                "",
                f"Columns differ.\nExpected: {list(expected_df.columns)}\nActual:   {list(actual_df.columns)}",
            )
        # Shape check
        if actual_df.shape != expected_df.shape:
            return (
                1,
                "",
                f"Shape differs.\nExpected: {expected_df.shape}\nActual:   {actual_df.shape}",
            )
        # Full equality check
        try:
            import pandas.testing as pdt
            pdt.assert_frame_equal(actual_df, expected_df, check_dtype=True, check_like=False, obj="parse")
        except AssertionError as e:
            return (1, "", f"DataFrames differ:\n{e}")
        return (0, "Built-in evaluation passed.", "")
    except Exception as e:
        return (1, "", f"Built-in evaluation error: {e}")

def extract_signal(stdout: str, stderr: str) -> str:
    text = "\n".join([(stderr or ""), (stdout or "")]).strip()
    if not text:
        return "No test output."
    lines: list[str] = []
    for m in re.findall(r"(ImportError:.*)", text):
        lines.append(m)
    m = re.search(r"Columns differ[^\n]*\nExpected:\s*(\[[^\n]*?\])\nActual:\s*(\[[^\n]*?\])", text, re.S)
    if m:
        lines.append(f"Expected columns: {m.group(1)}")
        lines.append(f"Actual columns: {m.group(2)}")
    m = re.search(r"Shape differs[^\n]*\nExpected:\s*\((\d+),\s*(\d+)\)\nActual:\s*\((\d+),\s*(\d+)\)", text)
    if m:
        e0, e1, a0, a1 = m.groups()
        lines.append(f"Expected shape: ({e0},{e1}) vs Actual: ({a0},{a1})")
    if not lines:
        lines.append("\n".join(text.splitlines()[-60:]))
    return "\n".join(lines)

def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    return m.group(1).strip() if m else text.strip()

def _read_text_if_exists(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def _load_examples() -> str:
    """Load optional few-shot examples to guide the LLM."""
    paths = [
        EXAMPLE_DIR / "camelot_example.py",
        EXAMPLE_DIR / "pdfplumber_example.py",
    ]
    blocks = []
    for p in paths:
        if p.exists():
            src = _read_text_if_exists(p)
            if src.strip():
                blocks.append(f"Example: {p.name}\n```python\n{src}\n```")
    return "\n\n".join(blocks)

def _baseline_extract_to_df(pdf_path: str, expected_cols: list[str]) -> pd.DataFrame:
    """
    Heuristic extractor to produce an 'expected' DataFrame from the PDF without using the generated parser.
    Tries camelot (stream, lattice) then pdfplumber; normalizes headers and numeric fields to expected_cols.
    """
    frames: list[pd.DataFrame] = []
    # 1) Camelot stream
    try:
        import camelot
        try:
            t_stream = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            frames += [t.df for t in t_stream if not t.df.empty]
        except Exception:
            pass
        # 2) Camelot lattice (fallback)
        try:
            t_lattice = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
            frames += [t.df for t in t_lattice if not t.df.empty]
        except Exception:
            pass
    except ImportError:
        pass

    # 3) pdfplumber fallback if no frames
    if not frames:
        try:
            import pdfplumber
            rows = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            rows.append([("" if c is None else str(c).strip()) for c in row])
            if rows:
                df = pd.DataFrame(rows)
                frames.append(df)
        except Exception:
            pass

    if not frames:
        raise RuntimeError("Could not extract any tables from PDF to build expected CSV")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.replace(r"^\s*$", pd.NA, regex=True)
    raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
    raw = raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Detect header row: pick first non-empty row with all strings
    header_tokens = None
    for i in range(min(5, len(raw))):
        vals = raw.iloc[i].tolist()
        if all(isinstance(x, str) and x.strip() for x in vals):
            header_tokens = [str(c).strip() for c in vals]
            raw = raw.iloc[i + 1 :].reset_index(drop=True)
            raw.columns = header_tokens
            break

    # Remove repeated header rows
    if header_tokens is not None:
        dup_mask = raw.apply(
            lambda r: [("" if v is None else str(v).strip()) for v in r.tolist()] == header_tokens,
            axis=1,
        )
        if dup_mask.any():
            raw = raw.loc[~dup_mask].reset_index(drop=True)

    # Simple aliasing to expected
    aliases = {
        "Txn Date": "Date",
        "Transaction Date": "Date",
        "Date": "Date",
        "Narration": "Description",
        "Description": "Description",
        "Withdrawal Amount": "Debit Amt",
        "Withdrawal": "Debit Amt",
        "Debit": "Debit Amt",
        "Deposit Amount": "Credit Amt",
        "Deposit": "Credit Amt",
        "Credit": "Credit Amt",
        "Closing Balance": "Balance",
        "Balance Amount": "Balance",
        "Balance": "Balance",
        "Amount": "Amount",
    }
    rename_map = {c: aliases.get(str(c).strip(), c) for c in raw.columns}
    df = raw.rename(columns=rename_map)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[expected_cols]

    # Normalize types
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    for col in [c for c in ["Debit Amt", "Credit Amt", "Amount", "Balance"] if c in df.columns]:
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


def ensure_expected_csv(target: str, pdf_path: str, csv_path: str, expected_cols: list[str]) -> str:
    """
    Creates data/{target}/result.csv when missing by running a baseline extractor.
    Returns the CSV path (existing or newly created).
    """
    out_path = csv_path
    if not out_path:
        out_path = str((PROJECT_ROOT / "data" / target / "result.csv").resolve())

    p = Path(out_path)
    if p.exists():
        return out_path

    p.parent.mkdir(parents=True, exist_ok=True)
    _log(f"No CSV found; creating baseline expected CSV at {out_path}")
    df = _baseline_extract_to_df(pdf_path, expected_cols)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

# ---------- LLM (Gemini via LangChain) ----------

SYS_PROMPT = (
    "You are an expert Python engineer. Generate or refine a parser module for bank statement PDFs.\n"
    "Requirements:\n"
    "- Provide a complete Python module exposing `parse(pdf_path: str) -> pandas.DataFrame`.\n"
    "- Use either camelot or pdfplumber; handle missing libs gracefully.\n"
    "- Detect header once; drop repeated header rows across pages.\n"
    "- Normalize date; clean numeric columns (commas/currency/Dr/Cr -> numbers).\n"
    "- Output EXACT columns and order from EXPECTED_COLS.\n"
    "- Return a DataFrame with those columns; create missing ones as NA.\n"
    "- Only output code (no extra commentary)."
    "- If a shape problem occur, probably the parser in taking the heading from the PDF 2 times. Take care of that."
)

def make_model(model_name: str = "gemini-1.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model_name)

def llm_generate(model: ChatGoogleGenerativeAI, expected_cols: list[str], target: str, pdf_path: str) -> str:
    examples = _load_examples()
    examples_note = ("\n\nUse the following examples as reference patterns:\n"
                     f"{examples}") if examples else ""
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=(
            f"Target bank: {target}\n"
            f"Sample PDF path: {pdf_path}\n"
            f"EXPECTED_COLS = {json.dumps(expected_cols, ensure_ascii=False)}\n"
            "Generate the parser module now."
            f"{examples_note}"
        )),
    ]
    ai = model.invoke(messages)
    code = extract_code_block(ai.content if isinstance(ai, AIMessage) else str(ai))
    if "EXPECTED_COLS" not in code:
        code = f'EXPECTED_COLS = {json.dumps(expected_cols, ensure_ascii=False)}\n' + code
    return code

def llm_refine(model: ChatGoogleGenerativeAI, source_code: str, failure_signal: str, expected_cols: list[str], target: str) -> str:
    examples = _load_examples()
    examples_note = ("\n\nUse the following examples as reference patterns:\n"
                     f"{examples}") if examples else ""
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=(
            f"Tests failed for target '{target}'. Failure signal:\n"
            f"{failure_signal}\n\n"
            "Refine the parser code to fix the issues while keeping the contract:\n"
            "1) Keep parse(pdf_path) signature\n"
            "2) Output EXACT columns in EXPECTED_COLS and order\n"
            "3) Handle missing libraries gracefully\n"
            "4) Avoid duplicating header rows across pages\n\n"
            "Current code:\n"
            f"```python\n{source_code}\n```"
            f"{examples_note}"
        )),
    ]
    ai = model.invoke(messages)
    code = extract_code_block(ai.content if isinstance(ai, AIMessage) else str(ai))
    if "EXPECTED_COLS" not in code:
        code = f'EXPECTED_COLS = {json.dumps(expected_cols, ensure_ascii=False)}\n' + code
    return code

# ---------- State + Nodes (LangGraph) ----------

class AgentState(TypedDict, total=False):
    # planning
    target: str
    pdf_path: str
    csv_path: str
    parser_path: str
    expected_cols: list[str]
    # llm memory
    messages: list  # optional chat log if you want to persist; unused for control
    # artifacts
    parser_code: str
    # testing
    test_path: Optional[str]
    test_exit: int
    test_stdout: str
    test_stderr: str
    failure_signal: str
    # control
    attempts: int
    done: bool

model = make_model()

def node_plan(state: AgentState) -> AgentState:
    _log("Planning...")
    info = plan(state["target"])
    state["pdf_path"] = info["pdf_path"]
    state["csv_path"] = info["csv_path"]
    state["parser_path"] = info["parser_path"]
    if info["csv_path"]:
        state["expected_cols"] = read_csv_header(info["csv_path"])
    else:
        _log("No CSV found; using default EXPECTED_COLS.")
        state["expected_cols"] = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]
    state["attempts"] = 0
    state["done"] = False
    _log(f"Target: {state['target']}")
    _log(f"PDF: {state['pdf_path']}")
    _log(f"CSV: {state['csv_path'] or '(none)'}")
    _log(f"Parser path: {state['parser_path']}")
    _log(f"Expected columns: {state['expected_cols']}")
    return state

def node_generate(state: AgentState) -> AgentState:
    # Log which examples are available (if any)
    available = [p.name for p in (EXAMPLE_DIR / "camelot_example.py", EXAMPLE_DIR / "pdfplumber_example.py") if p.exists()]
    if available:
        _log(f"Few-shot examples detected: {', '.join(available)}")
    else:
        _log("No few-shot examples found (prompts/examples/*.py). Proceeding without examples.")

    _log("Generating initial parser code with Gemini...")
    code = llm_generate(model, state["expected_cols"], state["target"], state["pdf_path"])
    write_text(state["parser_path"], code)
    state["parser_code"] = code
    _log(f"Wrote parser to {state['parser_path']} ({len(code)} bytes)")
    return state

def node_test(state: AgentState) -> AgentState:
    # Prefer pytest when a test path exists; otherwise use built-in evaluator for any target.
    test_path = state.get("test_path")
    candidate = PROJECT_ROOT / test_path if test_path else None

    if candidate and candidate.exists():
        _log(f"Running tests with pytest: {test_path}")
        code, out, err = run_pytest(test_path)
    else:
        # Ensure CSV exists (auto-create if missing), then run built-in eval
        try:
            csv_path = state.get("csv_path", "")
            csv_path = ensure_expected_csv(state["target"], state["pdf_path"], csv_path, state["expected_cols"])
            state["csv_path"] = csv_path
            _log("Running built-in evaluation (CSV comparison).")
            code, out, err = run_builtin_eval(state["target"], state["pdf_path"], csv_path)
        except Exception as e:
            code, out, err = 1, "", f"Built-in evaluation error: {e}"

    state["test_exit"] = code
    state["test_stdout"] = out
    state["test_stderr"] = err
    _log(f"Pytest exit code: {code}")
    if code != 0:
        _log("Test stderr (tail):")
        print(_tail(err, 40))
    return state

def node_observe(state: AgentState) -> AgentState:
    if state.get("test_exit", 1) == 0:
        _log("All tests passed ✅")
        state["failure_signal"] = ""
        state["done"] = True
        return state
    _log("Analyzing failures...")
    state["failure_signal"] = extract_signal(state.get("test_stdout", ""), state.get("test_stderr", ""))
    _log("Failure signal (summary):")
    print(_tail(state["failure_signal"], 60))
    state["done"] = False
    return state

def node_refine(state: AgentState) -> AgentState:
    if state.get("done"):
        return state
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        _log(f"Max attempts reached ({MAX_ATTEMPTS}). Stopping.")
        state["done"] = True
        return state
    state["attempts"] = state.get("attempts", 0) + 1
    _log(f"Refining parser with Gemini (attempt {state['attempts']}/{MAX_ATTEMPTS})...")
    current = Path(state["parser_path"]).read_text(encoding="utf-8")
    revised = llm_refine(model, current, state["failure_signal"], state["expected_cols"], state["target"])
    if revised and revised != current:
        write_text(state["parser_path"], revised)
        state["parser_code"] = revised
        _log(f"Updated parser. Size: {len(current)} -> {len(revised)} bytes")
    else:
        _log("Refinement produced no changes.")
    return state

# ---------- Build Graph with memory ----------

memory = MemorySaver()
graph = StateGraph(AgentState)
graph.add_node("plan", node_plan)
graph.add_node("generate", node_generate)
graph.add_node("test", node_test)
graph.add_node("observe", node_observe)
graph.add_node("refine", node_refine)

graph.add_edge(START, "plan")
graph.add_edge("plan", "generate")
graph.add_edge("generate", "test")
graph.add_edge("test", "observe")

def route(state: AgentState):
    if state.get("done") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return END
    return "refine"

graph.add_conditional_edges("observe", route, {"refine": "refine", END: END})
graph.add_edge("refine", "test")

app = graph.compile(checkpointer=memory)

# ---------- CLI ----------

def main():
    global MAX_ATTEMPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Bank key, e.g., icici")
    ap.add_argument("--tests", default="", help="Path to pytest file (empty to auto/none)")
    ap.add_argument("--thread", default="agent-run-1", help="LangGraph thread_id for memory")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, help="Max refinement attempts")
    args = ap.parse_args()

    # Auto-select ICICI test if target=icici and tests not provided
    test_path = args.tests
    icici_test = PROJECT_ROOT / "tests" / "test_icici_parser.py"
    if not test_path and args.target.lower() == "icici" and icici_test.exists():
        test_path = "tests/test_icici_parser.py"

    MAX_ATTEMPTS = args.max_attempts

    _log(f"Starting agent loop for target='{args.target}' (max_attempts={MAX_ATTEMPTS})")
    _log(f"Using memory thread: {args.thread}")
    _log(f"Tests path: {test_path or '(none)'}")

    init: AgentState = {
        "target": args.target,
        "test_path": test_path,
    }
    final = app.invoke(init, config={"configurable": {"thread_id": args.thread}})
    code = final.get("test_exit", 1)
    _log(f"Exit: {code}")
    if code != 0:
        _log("Failure signal:")
        print(final.get("failure_signal", ""))
        print(final.get("test_stdout", ""))
        print(final.get("test_stderr", ""))
        sys.exit(code)
    _log("Tests passed ✅")

if __name__ == "__main__":
    main()