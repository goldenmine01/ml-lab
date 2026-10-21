"""Print Python version and the installed version of every package in requirements.txt.

Usage (from repo root):  python lab01/src/print_versions.py
Output is printed and also saved to lab01/results/versions.txt
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = LAB_DIR.parent
RESULTS = LAB_DIR / "results"


def main() -> None:
    lines = [
        f"Python {sys.version.split()[0]} ({platform.python_implementation()})",
        f"Platform: {platform.platform()}",
        f"Machine: {platform.machine()} | Processor: {platform.processor() or 'n/a'}",
        "",
    ]
    req = REPO_DIR / "requirements.txt"
    for raw in req.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        name, _, pinned = raw.partition("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"
        flag = "" if installed == pinned else f"   <-- pinned {pinned}"
        lines.append(f"{name:<20} {installed}{flag}")

    text = "\n".join(lines)
    print(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "versions.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
