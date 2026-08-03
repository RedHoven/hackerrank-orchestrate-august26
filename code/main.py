#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from build_output_review import main as build_output_review
from build_sample_review import main as build_sample_review
from router import OUTPUT_FIELDS, ROUTERS, route_rows


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=int, choices=sorted(ROUTERS), default=3)
    parser.add_argument("--input", type=Path, default=DATA / "messages.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output.csv")
    parser.add_argument("--skip-review", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        predictions = route_rows(list(csv.DictReader(handle)), DATA, router_version=args.router)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Router {args.router} wrote {len(predictions)} predictions to {args.output}")
    if not args.skip_review and args.output.resolve() == ROOT / "output.csv" and args.input.resolve() == DATA / "messages.csv":
        build_output_review()
        build_sample_review()
        print(f"Review data and page refreshed at {ROOT / 'review.csv'}")


if __name__ == "__main__":
    main()
