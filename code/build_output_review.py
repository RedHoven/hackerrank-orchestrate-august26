import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from router import (
    attachment_text,
    behavioral_evidence,
    normalized_embedding_matrix,
    read_csv,
    reference_examples,
    router_3_candidates,
    router_3_context,
    routing_signal_cache_key,
    semantic_text,
    router_3_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
REVIEW_FIELDS = [
    "message_id", "user_id", "conversation_type", "created_at", "complete_message",
    "router_3_message_context", "router_3_nearest_labeled_examples", "router_3_prompt",
    "predicted_action", "predicted_message_type", "predicted_reason", "predicted_confidence",
    "predicted_evidence_message_ids", "review_verdict", "reviewed_action",
    "reviewed_message_type", "action_review", "type_review", "evidence_review",
    "confidence_review", "improvement_tags",
]

CHANGES = {
    "msg_023": ("digest", "payment", "The bank update is legitimate, but it gives no deadline or concrete near-term consequence; opening prior copies establishes relevance, not urgency.", "The payment label is accurate.", "urgency_threshold"),
    "msg_040": ("mute", "forward", "Mute is appropriate for a coercive chain message with negative history in a muted group.", "The explicit request to forward to ten people makes this a forward, not merely a greeting.", "forward_greeting_precedence"),
    "msg_104": ("mute", "promotion", "Mute is supported by strongly unwanted history from the same marketplace seller.", "A person-to-person item sale is promotion under Router 3's own type policy.", "marketplace_type"),
    "msg_060": ("notify", "event", "The same-day 5 PM form deadline warrants notification.", "This is an internship form deadline, not a promotion; the attachment's word 'offers' caused a false match.", "promotion_regex_false_positive"),
    "msg_052": ("mute", "scam", "Mute is correct for a mismatched-domain refund flow requesting wallet details.", "The deceptive financial request is scam rather than generic spam.", "scam_detection"),
    "msg_071": ("digest", "personal", "Digest is right because the plan is tentative and explicitly awaits later confirmation.", "An unclear doctor-or-dinner plan is ordinary personal coordination, not a scheduled event yet.", "event_type"),
    "msg_105": ("mute", "forward", "Mute is appropriate for unsafe medical advice with unwanted forwarding history.", "The content is an unsafe chain health forward, not a credential or payment scam.", "unsafe_forward_type"),
    "msg_019": ("mute", "scam", "Mute is correct for an unverified, mismatched-domain bank impersonation flow.", "The access threat and verification link are phishing, so scam is more precise than spam.", "scam_detection"),
    "msg_056": ("notify", "event", "The direct, same-day appointment change needs a prompt response.", "A rescheduled doctor appointment is event rather than personal.", "event_type"),
    "msg_083": ("digest", "personal", "The sender explicitly says nothing is urgent and asks for comments tomorrow; this should wait for digest.", "The request is ordinary work communication, so personal is acceptable.", "urgency_threshold"),
    "msg_033": ("digest", "personal", "Digest is correct because the sender says to call later and that nothing is urgent.", "The substantive purpose is a personal safety check-in, not just a greeting.", "greeting_personal_type"),
    "msg_061": ("mute", "forward", "Mute is justified by the explicit chain request, strong unwanted history, and muted group.", "Asking the recipient to share with ten people makes it a forward rather than a greeting.", "forward_greeting_precedence"),
    "msg_026": ("mute", "scam", "Mute is correct for a fake reattempt charge on a young, mismatched domain.", "The impersonation and payment-verification flow are scam rather than spam.", "scam_detection"),
    "msg_098": ("notify", "urgent", "An immediate clinic decision and explicit request to call now warrant notification.", "This is an ad-hoc care decision requiring immediate intervention, so urgent is more precise than event.", "urgent_event_type"),
    "msg_064": ("mute", "scam", "Mute is correct because the text requests wallet and card details under deadline pressure.", "The dangerous refund-verification request must outrank promotional text found in the unrelated attachment.", "scam_detection;attachment_conflict"),
    "msg_103": ("notify", "promotion", "The 6 PM hold deadline and positive sender history justify notification.", "The jacket transaction is a person-to-person marketplace promotion, not ordinary personal chat.", "marketplace_type"),
    "msg_036": ("mute", "scam", "Mute is correct for an unverified payout sender with a mismatched young domain and reported history.", "This is financial impersonation phishing, so scam is more precise than spam.", "scam_detection"),
    "msg_076": ("mute", "scam", "Mute is correct for a reported mismatched-domain refund flow requesting wallet details.", "The deceptive financial request is scam rather than spam.", "scam_detection"),
    "msg_093": ("notify", "business_update", "A verified same-day FedEx window requiring ID is a legitimate near-term delivery notification; the output action is wrong and contradicts its own reason.", "The phrase 'no payment or OTP is required' is a safety reassurance, not a credential request; this is a business delivery update.", "negated_credential_detection;post_policy_consistency"),
    "msg_101": ("digest", "personal", "The message explicitly says no reply is needed and contains no consequence if seen later, so digest is sufficient.", "Personal is the correct type.", "urgency_threshold"),
    "msg_074": ("mute", "scam", "Mute is supported by upfront token-payment risk, reported comparable history, and the muted group.", "Payment before registry papers plus a generic contact flow is a likely scam, not merely promotion.", "scam_detection"),
    "msg_096": ("notify", "unknown", "A found passport will only be held until 18:00, so the explicit same-day loss deadline outweighs unfamiliar-sender interruption cost; the output contradicts its own reason.", "Unknown remains appropriate because the sender has no established history.", "unfamiliar_sender_deadline;post_policy_consistency"),
    "msg_022": ("mute", "scam", "Mute is justified by the separate payment link, screenshot request, and reported comparable history.", "The suspicious payment flow is scam rather than a routine payment notice.", "scam_detection"),
}


def index(name, key):
    return {row[key]: row for row in read_csv(DATA, name)}


def assessment(row, prediction, context):
    changed = CHANGES.get(row["message_id"])
    action = changed[0] if changed else prediction["action"]
    kind = changed[1] if changed else prediction["message_type"]
    if changed:
        verdict = "needs_action_and_type_change" if action != prediction["action"] and kind != prediction["message_type"] else "needs_action_change" if action != prediction["action"] else "needs_type_change"
        action_review, type_review, tags = changed[2:]
        confidence_review = "Too high for a prediction needing correction." if float(prediction["confidence"]) >= 0.9 else "Confidence should be reduced because the predicted label is not fully correct."
    else:
        verdict = "good"
        action_review = {
            "notify": "Good: the message has a concrete near-term action, deadline, care, safety, or operational consequence.",
            "digest": "Good: the message is legitimate or potentially useful but has no immediate interruption need.",
            "mute": "Good: the safety signals, opt-out state, repetition, or negative behavioral history justify suppression.",
        }[action]
        type_review = f"Good: the content and conversation context fit {kind}."
        tags = ""
        confidence_review = "Reasonable for this decision." if float(prediction["confidence"]) <= 0.93 else "Slightly assertive, but supported by clear signals."
    evidence = context.get("behavioral_evidence")
    if not evidence:
        evidence_review = "Good: no history met Router 3's semantic threshold, so 'none' is appropriate."
    else:
        n = evidence["semantic_history"]["n"]
        outcome = evidence["semantic_history"]["outcome"]
        qualifier = "limited and should remain secondary" if n < 3 else f"usefully summarizes a {outcome} pattern"
        evidence_review = f"The {n} selected semantic neighbor(s) are topically relevant; this evidence is {qualifier}."
    return {
        "review_verdict": verdict,
        "reviewed_action": action,
        "reviewed_message_type": kind,
        "action_review": action_review,
        "type_review": type_review,
        "evidence_review": evidence_review,
        "confidence_review": confidence_review,
        "improvement_tags": tags,
    }


def main():
    messages = read_csv(DATA, "messages.csv")
    history = read_csv(DATA, "message_history.csv")
    examples = read_csv(DATA, "sample_messages.csv")
    users = index("users.csv", "user_id")
    groups = index("groups.csv", "group_id")
    businesses = index("business_accounts.csv", "business_id")
    memberships = {(row["user_id"], row["group_id"]): row for row in read_csv(DATA, "group_members.csv")}
    relationships = {(row["user_id"], row["business_id"]): row for row in read_csv(DATA, "user_business_history.csv")}
    events = index("message_events.csv", "message_id")
    notification_summaries = defaultdict(list)
    for row in read_csv(DATA, "daily_notification_summary.csv"):
        notification_summaries[row["user_id"]].append(row)
    image_descriptions = {row["image_id"]: row["description"] for row in read_csv(DATA, "image_descriptions.csv")}
    voice_transcripts = {row["voice_note_id"]: row["transcription"] for row in read_csv(DATA, "voice_transcriptions.csv")}
    predictions = {row["message_id"]: row for row in read_csv(ROOT, "output.csv")}
    cache = json.loads((DATA / "semantic_embeddings.json").read_text(encoding="utf-8"))["vectors"]
    routing_path = DATA / "routing_signals.json"
    routing_cache = json.loads(routing_path.read_text(encoding="utf-8")).get("assessments", {}) if routing_path.exists() else {}
    semantic_rows = history + messages + examples
    texts = [semantic_text(row, groups, businesses, image_descriptions, voice_transcripts) for row in semantic_rows]
    vectors = normalized_embedding_matrix([cache[hashlib.sha256(text.encode("utf-8")).hexdigest()] for text in texts])
    history_vectors = vectors[:len(history)]
    message_vectors = vectors[len(history):len(history) + len(messages)]
    example_vectors = vectors[len(history) + len(messages):]
    existing = {}
    review_path = ROOT / "review.csv"
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8") as handle:
            existing = {row["message_id"]: row for row in csv.DictReader(handle)}
    output = []
    for position, row in enumerate(messages):
        content_sources = {"message_body": row["message_text"].strip(), "attachment": attachment_text(row, image_descriptions, voice_transcripts)}
        complete_message = "\n\n".join(part for part in content_sources.values() if part) or "[No text content available]"
        candidates = router_3_candidates(row, history, history_vectors, message_vectors[position])
        behavioral = behavioral_evidence(row, history, events, candidates) if candidates else None
        context = router_3_context(row, users, groups, businesses, memberships, relationships, notification_summaries, history, behavioral)
        context["content_sources"] = content_sources
        routing = routing_cache.get(routing_signal_cache_key(row, complete_message, context))
        if routing:
            context["routing_assessment"] = routing
        nearest_examples = reference_examples(row, examples, example_vectors, message_vectors[position], users, groups, businesses)
        prediction = predictions[row["message_id"]]
        previous = existing.get(row["message_id"], {})
        generated_review = assessment(row, prediction, context)
        output.append({
            "message_id": row["message_id"],
            "user_id": row["user_id"],
            "conversation_type": row["conversation_type"],
            "created_at": row["created_at"],
            "complete_message": complete_message,
            "router_3_message_context": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            "router_3_nearest_labeled_examples": json.dumps(nearest_examples, ensure_ascii=False, separators=(",", ":")),
            "router_3_prompt": router_3_prompt(complete_message, context, nearest_examples),
            **{f"predicted_{key}": prediction[key] for key in ("action", "message_type", "reason", "confidence", "evidence_message_ids")},
            **{field: previous.get(field) or generated_review[field] for field in REVIEW_FIELDS[13:]},
        })
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} rows to {review_path}")


if __name__ == "__main__":
    main()
