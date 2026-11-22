"""Fail when requirements.txt contains an unpinned or malformed dependency."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
PIN = re.compile(r"^[A-Za-z0-9_.-]+==[^\s;]+$")


def main() -> int:
    entries = [line.strip() for line in REQUIREMENTS.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    invalid = [line for line in entries if not PIN.fullmatch(line)]
    if not entries or invalid:
        print("requirements.txt must contain only exact package==version pins.")
        for entry in invalid:
            print(f"  invalid: {entry}")
        return 1
    print(f"requirements pin check passed ({len(entries)} exact pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
