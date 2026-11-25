import re
import resource
import subprocess
import tempfile
from pathlib import Path


def last_box(text):
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start:i]
    return None


def numeric(text):
    from fractions import Fraction
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = re.sub(r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+(?:\.\d*)?)?", text)
    if not matches:
        return None
    try:
        return str(Fraction(matches[-1]))
    except (ValueError, ZeroDivisionError):
        return None


def code_candidate(text, row):
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    code = fences[-1] if fences else text
    if re.search(r"^\s*(?:async\s+)?def\s+"+re.escape(row["entry_point"])+r"\s*\(",
                 code, flags=re.M):
        return code
    return row["prompt"]+code


def score(text, task):
    dataset = task["dataset"]
    if dataset == "gsm8k":
        extracted = numeric(last_box(text) or text)
        ref = numeric(task["reference"])
        return {"correct": extracted is not None and extracted == ref,
                "extracted": extracted, "reference": ref, "metric": "numeric_exact_match"}
    if dataset == "math500":
        from math_verify import parse, verify
        boxed = last_box(text)
        prediction = parse("\\boxed{"+boxed+"}") if boxed is not None else []
        gold = parse("\\boxed{"+task["reference"]+"}")
        return {"correct": bool(gold and prediction and verify(gold, prediction)),
                "extracted": str(prediction), "reference": str(gold),
                "metric": "math_verify_0.8.0_last_box"}
    row = task["source"]
    code = code_candidate(text, row)
    test = code+"\n\n"+row["test"]+"\ncheck("+row["entry_point"]+")\n"
    with tempfile.TemporaryDirectory(prefix="fixed-baseline-code-") as folder:
        path = Path(folder)/"check.py"
        path.write_text(test)
        # No project or model data is mounted; every visible host directory is
        # read-only, /tmp is private, and all networking is unshared.
        argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64", "--proc", "/proc",
                "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", folder, "/work",
                "--chdir", "/work", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                "/usr/bin/python3", "-I", "/work/check.py"]

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024, 1024*1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=10,
                                    preexec_fn=limits, env={"PATH": "/usr/bin:/bin"})
            if "bwrap:" in result.stderr:
                raise RuntimeError("Code evaluation sandbox unavailable: "+result.stderr)
            return {"correct": result.returncode == 0, "metric": "humaneval_base_pass1",
                    "exit_code": result.returncode, "stderr": result.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"correct": False, "metric": "humaneval_base_pass1", "timeout": True}
