"""
chatbot.py
Handles the "AI Data Analyst Chatbot":
  1) The user asks a question (English or Arabic) about the data.
  2) The local model (via Ollama) converts the question into pandas code.
  3) Python actually executes the code on the data.
  4) The model explains the result in plain language.
"""

import re
import io
import contextlib
import pandas as pd
import numpy as np
import plotly.express as px

from modules.llm_client import ask_ollama

CODE_SYSTEM_PROMPT = """
You are an expert Python data analyst. You have a DataFrame called df with the columns and types given below.
Your task: write pandas/plotly Python code ONLY (no explanation) that answers the user's question.

Mandatory rules:
- Only use: df, pd, np, px (plotly.express) - already available, do not import anything.
- If the question needs a number or table: store the output in a variable called result.
- If the question needs a chart: store the figure in a variable called fig using px only.
- Strictly forbidden: import, exec, eval, open, os, sys, subprocess, __, globals, locals, input, write, delete, drop (on disk), any file/network/system operation.
- Do not permanently modify df (e.g. df.drop(inplace=True)) - use a copy if needed.
- Output code only, no ```python fences or extra text.
"""

FORBIDDEN_PATTERNS = [
    r"\bimport\b", r"\bexec\b", r"\beval\b", r"\bopen\b", r"\bos\.", r"\bsys\.",
    r"subprocess", r"__", r"\bglobals\b", r"\blocals\b", r"\binput\b",
    r"\.to_csv\(", r"\.to_excel\(", r"\.to_sql\(", r"\bdel\b",
    r"inplace\s*=\s*True", r"requests\.", r"socket",
]


def _build_schema_text(df: pd.DataFrame) -> str:
    lines = ["Available columns in df:"]
    for col in df.columns:
        lines.append(f"- {col} ({df[col].dtype})")
    lines.append("\nSample of the first 3 rows:")
    lines.append(df.head(3).to_string())
    return "\n".join(lines)


def _is_code_safe(code: str) -> bool:
    return not any(re.search(p, code) for p in FORBIDDEN_PATTERNS)


def _clean_code(raw: str) -> str:
    code = raw.strip()
    code = re.sub(r"^```(python)?", "", code, flags=re.IGNORECASE).strip()
    code = re.sub(r"```$", "", code).strip()
    return code


def generate_code(question: str, df: pd.DataFrame, model: str = "llama3.1") -> str:
    schema_text = _build_schema_text(df)
    prompt = f"{schema_text}\n\nUser question: {question}\n\nWrite Python code only to answer this question."
    raw = ask_ollama(prompt, model=model, system=CODE_SYSTEM_PROMPT, temperature=0.1)
    return _clean_code(raw)


def execute_code(code: str, df: pd.DataFrame):
    """
    Executes the generated code in a restricted (sandboxed) environment and returns
    (result, fig, error, stdout)
    """
    if not _is_code_safe(code):
        return None, None, "⚠️ Code execution was rejected because it contains disallowed operations (security).", ""

    safe_globals = {
        "__builtins__": {
            "len": len, "range": range, "sum": sum, "min": min, "max": max,
            "round": round, "sorted": sorted, "list": list, "dict": dict,
            "set": set, "tuple": tuple, "str": str, "int": int, "float": float,
            "bool": bool, "abs": abs, "enumerate": enumerate, "zip": zip,
            "print": print,
        },
        "pd": pd,
        "np": np,
        "px": px,
        "df": df.copy(),
    }
    local_vars = {}
    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals, local_vars)
        result = local_vars.get("result", None)
        fig = local_vars.get("fig", None)
        return result, fig, None, stdout_capture.getvalue()
    except Exception as e:
        return None, None, f"⚠️ An error occurred while running the analysis: {e}", stdout_capture.getvalue()


def explain_result(question: str, result, model: str = "llama3.1") -> str:
    result_text = str(result)[:3000]  # cap the text length
    prompt = (
        f"User question: {question}\n\n"
        f"Actual analysis result computed on the data:\n{result_text}\n\n"
        "Explain this result to the user in one or two simple, clear sentences, without rewriting the numbers as a table."
    )
    system = "You are a data analyst explaining a real analysis result to a non-technical business owner, concisely and clearly, in English."
    return ask_ollama(prompt, model=model, system=system, temperature=0.3)


def answer_question(question: str, df: pd.DataFrame, model: str = "llama3.1") -> dict:
    """
    Main function that ties all the steps together: generate code -> execute -> explain result.
    """
    code = generate_code(question, df, model=model)
    result, fig, error, stdout = execute_code(code, df)

    explanation = None
    if error is None and (result is not None or stdout):
        explanation = explain_result(question, result if result is not None else stdout, model=model)

    return {
        "code": code,
        "result": result,
        "fig": fig,
        "error": error,
        "stdout": stdout,
        "explanation": explanation,
    }
