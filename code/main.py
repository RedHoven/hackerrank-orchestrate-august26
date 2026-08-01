#!/usr/bin/env python3
import csv
from pathlib import Path

from router import OUTPUT_FIELDS, route_rows


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"


def main():
    with (DATA / "messages.csv").open(newline="", encoding="utf-8") as handle:
        predictions = route_rows(list(csv.DictReader(handle)), DATA)
    with (ROOT / "output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to {ROOT / 'output.csv'}")


if __name__ == "__main__":
    main()
