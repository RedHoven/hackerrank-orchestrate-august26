#!/usr/bin/env python3
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from router import audio_to_text, image_to_text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUTPUT = DATA / "voice_transcriptions.csv"
IMAGE_OUTPUT = DATA / "image_descriptions.csv"


def main():
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    client.responses
    with (DATA / "voice_notes.csv").open(newline="", encoding="utf-8") as handle:
        voices = list(csv.DictReader(handle))
    existing = {}
    if OUTPUT.exists():
        with OUTPUT.open(newline="", encoding="utf-8") as handle:
            existing = {row["voice_note_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for voice in voices:
        prior = existing.get(voice["voice_note_id"])
        if prior and prior["status"] == "ok":
            rows.append(prior)
            continue
        source = DATA / voice["file_path"]
        with tempfile.TemporaryDirectory() as temp_dir:
            normalized = Path(temp_dir) / f"{voice['voice_note_id']}.wav"
            try:
                subprocess.run(["ffmpeg", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", str(normalized)], capture_output=True, check=True)
                rows.append({"voice_note_id": voice["voice_note_id"], "transcription": audio_to_text(client, normalized), "status": "ok"})
            except Exception as error:
                rows.append({"voice_note_id": voice["voice_note_id"], "transcription": "[Voice transcript unavailable]", "status": f"error: {type(error).__name__}"})
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["voice_note_id", "transcription", "status"])
        writer.writeheader()
        writer.writerows(rows)
    with (DATA / "images.csv").open(newline="", encoding="utf-8") as handle:
        images = list(csv.DictReader(handle))
    existing_images = {}
    if IMAGE_OUTPUT.exists():
        with IMAGE_OUTPUT.open(newline="", encoding="utf-8") as handle:
            existing_images = {row["image_id"]: row for row in csv.DictReader(handle)}
    image_rows = []
    pending_images = []
    for image in images:
        prior = existing_images.get(image["image_id"])
        if prior and prior["status"] == "ok":
            image_rows.append(prior)
        else:
            pending_images.append(image)

    def describe(image):
        source = DATA / image["file_path"]
        try:
            return {"image_id": image["image_id"], "description": image_to_text(client, source), "status": "ok"}
        except Exception:
            with tempfile.TemporaryDirectory() as temp_dir:
                normalized = Path(temp_dir) / f"{image['image_id']}.png"
                try:
                    subprocess.run(["sips", "-s", "format", "png", str(source), "--out", str(normalized)], capture_output=True, check=True)
                    return {"image_id": image["image_id"], "description": image_to_text(client, normalized), "status": "ok"}
                except Exception:
                    try:
                        subprocess.run(["ffmpeg", "-y", "-i", str(source), str(normalized)], capture_output=True, check=True)
                        return {"image_id": image["image_id"], "description": image_to_text(client, normalized), "status": "ok"}
                    except Exception as error:
                        return {"image_id": image["image_id"], "description": "[Image description unavailable]", "status": f"error: {type(error).__name__}"}

    for start in range(0, len(pending_images), 4):
        with ThreadPoolExecutor(max_workers=4) as executor:
            image_rows.extend(executor.map(describe, pending_images[start:start + 4]))
        with IMAGE_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "description", "status"])
            writer.writeheader()
            writer.writerows(image_rows)
    print(f"Wrote {len(rows)} voice transcripts and {len(image_rows)} image descriptions")


if __name__ == "__main__":
    main()
