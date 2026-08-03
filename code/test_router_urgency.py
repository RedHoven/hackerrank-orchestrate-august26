import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from router import apply_origin_preference_veto, cached_routing_signals, routing_signal_cache_key


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        immediate = "avant 18h" in kwargs["input"]
        return SimpleNamespace(output_text=json.dumps({
            "safety_risk": "none",
            "user_relevance": "high" if immediate else "possible",
            "interruption_need": "immediate" if immediate else "digest",
            "active_transaction_evidence": "none",
            "primary_intent": "immediate_intervention" if immediate else "ordinary_communication",
            "time_window": "before 18:00" if immediate else "tomorrow before lunch",
            "rationale": "A same-day recovery deadline." if immediate else "The request can wait for a normal digest.",
        }))


class RoutingSignalTests(unittest.TestCase):
    def row(self, created_at="2026-07-31 12:38"):
        return {
            "user_id": "u_001",
            "conversation_type": "personal",
            "created_at": created_at,
            "forwarded_count": "0",
        }

    def test_cache_key_includes_time_context(self):
        message = "Please come before 18h."
        context = {"same_sender_history_count": 0}
        self.assertNotEqual(routing_signal_cache_key(self.row(), message, context), routing_signal_cache_key(self.row("2026-08-01 12:38"), message, context))

    def test_assessments_are_persisted_and_reused(self):
        client = SimpleNamespace(responses=FakeResponses())
        items = [
            (self.row(), "Votre passeport a ete trouve; venez avant 18h.", {"same_sender_history_count": 0}),
            (self.row(), "Please review this tomorrow before lunch. Nothing urgent.", {"same_sender_history_count": 0}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = cached_routing_signals(client, Path(directory), items)
            second = cached_routing_signals(client, Path(directory), items)
            self.assertEqual(first, second)
            self.assertEqual(len(client.responses.calls), 2)
            self.assertEqual(first[0]["interruption_need"], "immediate")
            self.assertEqual(first[1]["interruption_need"], "digest")

    def test_strongly_unwanted_commercial_origin_is_muted_without_transaction_evidence(self):
        decision = {"action": "notify", "message_type": "event", "reason": "Pickup deadline.", "confidence": 0.94}
        context = {"same_origin_preference": "strongly_unwanted", "routing_assessment": {"active_transaction_evidence": "none", "primary_intent": "commercial_offer"}}
        result = apply_origin_preference_veto(decision, context)
        self.assertEqual((result["action"], result["message_type"]), ("mute", "promotion"))
        self.assertEqual(apply_origin_preference_veto(decision, {"same_origin_preference": "strongly_unwanted", "routing_assessment": {"active_transaction_evidence": "strong", "primary_intent": "commercial_offer"}}), decision)


if __name__ == "__main__":
    unittest.main()
