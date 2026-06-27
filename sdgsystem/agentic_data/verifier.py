from __future__ import annotations

import ast
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def verify_rationale(item: dict[str, Any], code_path: Path, *, timeout: int = 30) -> dict[str, Any]:
    code_path = code_path.resolve()
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(str(item["rationale"]), encoding="utf-8")
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            cwd=str(code_path.parent),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.time() - started
        stdout = proc.stdout.strip()
        last_line = stdout.splitlines()[-1].strip() if stdout else ""
        expected = str(item["final_answer"]).strip()
        return {
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "observed_final_line": last_line,
            "expected_final_answer": expected,
            "matched": proc.returncode == 0 and answers_match(last_line, expected),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "returncode": None,
            "elapsed_seconds": time.time() - started,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "observed_final_line": "",
            "expected_final_answer": str(item["final_answer"]).strip(),
            "matched": False,
            "error": "timeout",
        }


def answers_match(observed: str, expected: str) -> bool:
    observed = str(observed).strip()
    expected = str(expected).strip()
    if observed == expected:
        return True
    if observed.lower() == expected.lower():
        return True
    try:
        return math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        pass
    try:
        return ast.literal_eval(observed) == ast.literal_eval(expected)
    except Exception:
        return False
