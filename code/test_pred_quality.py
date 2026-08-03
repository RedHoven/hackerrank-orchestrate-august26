#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict
from pathlib import Path

from router import router_0, router_1, router_2, router_3


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"

routers_to_test = ["router_3"]

routers = {
    "baseline": router_0,
    # "llm_msg": router_1,
    "llm_msg_evidence": router_2,
    "router_3": router_3,
}


def save_predictions(name, samples, predictions):
    fields = list(samples[0]) + ["predicted_action", "predicted_message_type", "predicted_reason", "predicted_confidence", "predicted_evidence_message_ids"]
    with (ROOT / f"sample_predictions_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            prediction = predictions[sample["message_id"]]
            writer.writerow({**sample, **{f"predicted_{key}": prediction[key] for key in ("action", "message_type", "reason", "confidence", "evidence_message_ids")}})


def print_matrix(title, actual, predicted):
    labels = sorted(set(actual) | set(predicted))
    matrix = defaultdict(Counter)
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    print(f"\n{title} confusion matrix (rows=actual, columns=predicted)")
    print("actual\\pred".ljust(18) + " ".join(label.ljust(16) for label in labels))
    for truth in labels:
        print(truth.ljust(18) + " ".join(str(matrix[truth][guess]).ljust(16) for guess in labels))


def evaluate(name, samples, router):
    predictions = {row["message_id"]: row for row in router(samples, DATA)}
    save_predictions(name, samples, predictions)
    predicted_rows = [predictions[row["message_id"]] for row in samples]
    action_actual = [row["action"] for row in samples]
    action_predicted = [row["action"] for row in predicted_rows]
    type_actual = [row["message_type"] for row in samples]
    type_predicted = [row["message_type"] for row in predicted_rows]
    action_accuracy = sum(a == b for a, b in zip(action_actual, action_predicted)) / len(samples)
    type_accuracy = sum(a == b for a, b in zip(type_actual, type_predicted)) / len(samples)
    joint_accuracy = sum(a == b and c == d for a, b, c, d in zip(action_actual, action_predicted, type_actual, type_predicted)) / len(samples)
    return action_actual, action_predicted, type_actual, type_predicted, (action_accuracy, type_accuracy, joint_accuracy)


def main():
    with (DATA / "sample_messages.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))

    router_results = []

    for router_name in routers_to_test:
        router_results.append(evaluate(router_name, samples, routers[router_name]))

    print(f"Samples: {len(samples)}")
    print("Router    Action accuracy   Type accuracy   Exact action + type")
    for i, router_name in enumerate(routers_to_test):
        print(f"{router_name}  {router_results[i][4][0]:>14.1%}  {router_results[i][4][1]:>14.1%}  {router_results[i][4][2]:>19.1%}")
        print_matrix(f"{router_name} action", router_results[i][0], router_results[i][1])
        print_matrix(f"{router_name} message type", router_results[i][2], router_results[i][3])


if __name__ == "__main__":
    main()
