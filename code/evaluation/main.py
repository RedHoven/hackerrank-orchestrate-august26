#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    result = subprocess.run([sys.executable, str(ROOT / "code" / "main.py")], cwd=ROOT, check=True)
    with (ROOT / "output.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    expected = list(csv.DictReader((ROOT / "dataset" / "messages.csv").open(newline="", encoding="utf-8")))
    assert list(rows[0]) == required if rows else True
    assert len(rows) == len(expected), (len(rows), len(expected))
    assert {row["message_id"] for row in rows} == {row["message_id"] for row in expected}
    assert all(row["action"] in {"notify", "digest", "mute"} for row in rows)
    assert all(0 <= float(row["confidence"]) <= 1 for row in rows)
    print(f"Validated {len(rows)} rows")


if __name__ == "__main__":
    main()
