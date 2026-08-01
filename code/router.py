from __future__ import annotations

import csv
import re
import shutil
import subprocess
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


OUTPUT_FIELDS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
WORD_RE = re.compile(r"[a-z0-9@]+")


def read_csv(data_dir, name):
    with (data_dir / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(value):
    try:
        return int(value or 0)
    except ValueError:
        return 0


def tokens(text):
    ignored = {"the", "and", "for", "with", "your", "this", "that", "from", "have", "are", "you", "now", "please", "message", "team"}
    return {word for word in WORD_RE.findall((text or "").lower()) if len(word) > 2 and word not in ignored}


def ocr_text(media_id, image_paths, cache):
    if media_id in cache:
        return cache[media_id]
    path = image_paths.get(media_id)
    if not path or not shutil.which("tesseract"):
        cache[media_id] = ""
        return ""
    try:
        result = subprocess.run(["tesseract", str(path), "stdout"], capture_output=True, text=True, timeout=25, check=False)
        cache[media_id] = result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        cache[media_id] = ""
    return cache[media_id]


def safety_risk(text, business):
    lower = text.lower()
    sensitive = bool(re.search(r"\b(otp|one.?time|password|pin|6.?digit|verification code|login code)\b", lower))
    pressure = bool(re.search(r"\b(blocked?|suspend|expire|immediately|urgent|within \d|today)\b", lower))
    suspicious = bool(re.search(r"\b(scan (this )?qr|crypto|gift card|claim reward|forward this to|ten people)\b", lower))
    domain_bad = business and business.get("official_domain") != business.get("domain_used_by_sender")
    return sensitive and (pressure or domain_bad or not business) or suspicious and pressure


def type_for(text, row, business):
    lower = text.lower()
    if safety_risk(text, business):
        return "scam"
    if re.search(r"\b(offer|sale|discount|coupon|promo|off\b|unsubscribe|buy now)\b", lower):
        return "promotion"
    if as_int(row.get("forwarded_count")) >= 3 or re.search(r"\b(fwd|forward|blessings|good morning)\b", lower):
        return "forward" if "forward" in lower or "fwd" in lower else "greeting"
    if re.search(r"\b(pay|payment|invoice|bill|refund|transaction|card)\b", lower):
        return "payment"
    if re.search(r"\b(order|delivery|booking|appointment|prescription|account|statement|service)\b", lower):
        return "business_update" if row["conversation_type"] == "business" else "event"
    if re.search(r"\b(today|tonight|tomorrow|minutes?|hours?|eod|deadline|emergency|urgent|asap|immediately)\b", lower):
        return "urgent"
    if re.search(r"\b(meeting|bus|school|form|circular|pickup|event|schedule|timing|register)\b", lower):
        return "event"
    if re.search(r"\b(hello|hi|good morning|good evening|hope|birthday)\b", lower):
        return "greeting"
    return "personal" if row["conversation_type"] == "personal" else "unknown"


def similarity(a, b):
    a_tokens, b_tokens = tokens(a), tokens(b)
    overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    sequence = SequenceMatcher(None, (a or "").lower()[:500], (b or "").lower()[:500]).ratio()
    return max(overlap, sequence * 0.65)


def route_rows(messages, data_dir):
    history = read_csv(data_dir, "message_history.csv")
    events = {row["message_id"]: row for row in read_csv(data_dir, "message_events.csv")}
    businesses = {row["business_id"]: row for row in read_csv(data_dir, "business_accounts.csv")}
    memberships = {(row["user_id"], row["group_id"]): row for row in read_csv(data_dir, "group_members.csv")}
    relationships = {(row["user_id"], row["business_id"]): row for row in read_csv(data_dir, "user_business_history.csv")}
    image_paths = {row["image_id"]: data_dir / row["file_path"] for row in read_csv(data_dir, "images.csv")}
    by_user = defaultdict(list)
    for row in history:
        by_user[row["user_id"]].append(row)
    ocr_cache = {}
    predictions = []

    for row in messages:
        business = businesses.get(row["business_id"])
        text = row["message_text"] or ""
        if row["media_type"] == "image":
            text = (text + " " + ocr_text(row["media_id"], image_paths, ocr_cache)).strip()
        kind = type_for(text, row, business)
        candidates = []
        for old in by_user[row["user_id"]]:
            context_bonus = 0.12 if old["business_id"] and old["business_id"] == row["business_id"] else 0
            context_bonus = max(context_bonus, 0.10 if old["group_id"] and old["group_id"] == row["group_id"] else 0)
            context_bonus = max(context_bonus, 0.08 if old["sender_user_id"] and old["sender_user_id"] == row["sender_user_id"] else 0)
            score = similarity(text, old["message_text"]) + context_bonus
            if score >= 0.28:
                candidates.append((score, old))
        candidates.sort(key=lambda item: item[0], reverse=True)
        evidence = candidates[:2]
        reactions = [events.get(old["message_id"], {}) for _, old in evidence]
        negative = sum(as_int(event.get("notification_dismissed")) + as_int(event.get("muted_after_message")) + 2 * as_int(event.get("message_reported")) for event in reactions)
        positive = sum(as_int(event.get("message_opened")) + 2 * as_int(event.get("message_replied")) for event in reactions)
        member = memberships.get((row["user_id"], row["group_id"]), {})
        relationship = relationships.get((row["user_id"], row["business_id"]), {})
        direct = f"@{row['user_id']}" in text.lower() or bool(re.search(r"\b(can you|please (?:call|reply|join)|need you)\b", text.lower()))
        urgent = kind == "urgent" or (kind == "event" and bool(re.search(r"\b(today|tonight|tomorrow|minutes?|hours?|early)\b", text.lower())))

        if safety_risk(text, business):
            action, reason = "mute", "The message requests sensitive verification or uses a suspicious, high-pressure flow."
        elif kind in {"forward", "greeting"} and (as_int(row["forwarded_count"]) >= 3 or negative > positive):
            action, reason = "mute", "This is a repeated forward or low-value greeting with negative engagement signals."
        elif kind == "promotion" and (relationship.get("promotions_opted_out_at") or relationship.get("allows_promotions") == "0" or negative > positive):
            action, reason = "mute", "The promotion conflicts with the user's opt-out or prior dismissal behavior."
        elif row["conversation_type"] == "group" and member.get("group_muted_by_user") == "1" and not direct and not urgent:
            action, reason = "mute", "The user has muted this group and the message has no direct or urgent need."
        elif row["conversation_type"] == "business" and business and business.get("verified") == "1" and relationship and (urgent or kind in {"payment", "business_update", "event"}):
            action, reason = "notify", "A verified business update matches the user's active relationship and may require timely action."
        elif direct and (urgent or row["conversation_type"] in {"personal", "group"}):
            action, reason = "notify", "The message directly asks this user for a timely response or action."
        elif urgent and row["conversation_type"] != "business":
            action, reason = "notify", "The message contains a time-sensitive operational request."
        elif kind in {"promotion", "greeting", "forward"}:
            action, reason = "digest", "The message is safe but non-urgent and can be reviewed later."
        elif row["conversation_type"] == "business" and relationship and business and business.get("verified") == "1":
            action, reason = "digest", "This is a legitimate business update, but it does not require an immediate interruption."
        else:
            action, reason = "digest", "The message appears safe and useful enough for a later digest rather than an interruption."
        if kind == "unknown" and action == "mute":
            kind = "spam"
        confidence = min(0.94, 0.68 + (0.09 if business and business.get("verified") == "1" else 0) + (0.07 if evidence else 0) + (0.08 if safety_risk(text, business) else 0) + (0.04 if direct else 0))
        predictions.append({"message_id": row["message_id"], "action": action, "message_type": kind, "reason": reason, "confidence": f"{confidence:.2f}", "evidence_message_ids": ";".join(old["message_id"] for _, old in evidence) or "none"})
    return predictions
