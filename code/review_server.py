#!/usr/bin/env python3
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from build_sample_review import main as build_review


ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS = ROOT / "router_3_corrections.json"
ACTIONS = {"notify", "digest", "mute"}
MESSAGE_TYPES = {"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"}


def message_ids():
    with (ROOT / "dataset" / "messages.csv").open(encoding="utf-8") as file:
        return {line.split(",", 1)[0] for line in file.read().splitlines()[1:]}


def load_corrections():
    try:
        data = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
        return data if isinstance(data.get("corrections"), list) else {"corrections": []}
    except FileNotFoundError:
        return {"corrections": []}


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        if self.path == "/api/corrections":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(load_corrections()).encode("utf-8"))
            return
        if self.path == "/":
            self.path = "/sample_router_review.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/corrections":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 4096:
                raise ValueError
            update = json.loads(self.rfile.read(size))
            message_id = update["message_id"]
            action = update.get("action")
            message_type = update.get("message_type")
            if message_id not in message_ids() or action not in ACTIONS | {None} or message_type not in MESSAGE_TYPES | {None}:
                raise ValueError
        except (ValueError, KeyError, json.JSONDecodeError):
            self.send_error(400, "Invalid correction")
            return
        data = load_corrections()
        corrections = {item["message_id"]: item for item in data["corrections"]}
        correction = {"message_id": message_id}
        if action:
            correction["action"] = action
        if message_type:
            correction["message_type"] = message_type
        if len(correction) == 1:
            corrections.pop(message_id, None)
        else:
            corrections[message_id] = correction
        data = {"corrections": sorted(corrections.values(), key=lambda item: item["message_id"])}
        temporary = CORRECTIONS.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(CORRECTIONS)
        self.send_response(204)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build_review()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    print(f"Review page: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
