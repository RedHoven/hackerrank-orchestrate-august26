#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict
from pathlib import Path

from router import route_rows


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"


def print_matrix(title, actual, predicted):
    labels = sorted(set(actual) | set(predicted))
    matrix = defaultdict(Counter)
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    print(f"\n{title} confusion matrix (rows=actual, columns=predicted)")
    print("actual\\pred".ljust(18) + " ".join(label.ljust(16) for label in labels))
    for truth in labels:
        print(truth.ljust(18) + " ".join(str(matrix[truth][guess]).ljust(16) for guess in labels))


def main():
    with (DATA / "sample_messages.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    predictions = {row["message_id"]: row for row in route_rows(samples, DATA)}
    predicted_rows = [predictions[row["message_id"]] for row in samples]
    action_actual = [row["action"] for row in samples]
    action_predicted = [row["action"] for row in predicted_rows]
    type_actual = [row["message_type"] for row in samples]
    type_predicted = [row["message_type"] for row in predicted_rows]
    action_accuracy = sum(a == b for a, b in zip(action_actual, action_predicted)) / len(samples)
    type_accuracy = sum(a == b for a, b in zip(type_actual, type_predicted)) / len(samples)
    joint_accuracy = sum(a == b and c == d for a, b, c, d in zip(action_actual, action_predicted, type_actual, type_predicted)) / len(samples)
    print(f"Samples: {len(samples)}")
    print(f"Action accuracy: {action_accuracy:.1%}")
    print(f"Message type accuracy: {type_accuracy:.1%}")
    print(f"Exact action + type accuracy: {joint_accuracy:.1%}")
    print_matrix("Action", action_actual, action_predicted)
    print_matrix("Message type", type_actual, type_predicted)


if __name__ == "__main__":
    main()
