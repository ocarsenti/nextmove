"""Tests for engine/job_extraction.py — the signals -> axes deterministic layer,
plus mocked-LLM tests for extract_signals()'s citation verification.

extract_signals() needs ANTHROPIC_API_KEY to make a real call — same boundary
as the rest of the codebase, so real calls aren't exercised here. But the
citation-verification logic around the call (does source_phrase actually
appear in the input text) is pure post-processing and fully testable with a
mocked Anthropic client — see TestCitationVerification.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.models import DetectedSignal
from ontology.registry import AxisRegistry
from engine.job_extraction import load_signal_library, signals_to_axes, extract_signals


class TestSignalLibrary(unittest.TestCase):

    def test_loads_signals(self):
        lib = load_signal_library()
        self.assertGreaterEqual(len(lib), 10)

    def test_every_effect_axis_is_real(self):
        registry = AxisRegistry()
        for s in load_signal_library():
            for axis_id in s.axis_effects:
                self.assertIsNotNone(registry.get(axis_id), f"signal {s.id} affects unknown axis {axis_id}")

    def test_every_effect_axis_is_non_meta(self):
        registry = AxisRegistry()
        meta_ids = {a.id for a in registry.meta_axes()}
        for s in load_signal_library():
            for axis_id in s.axis_effects:
                self.assertNotIn(axis_id, meta_ids, f"signal {s.id} affects a meta axis {axis_id}")

    def test_effect_values_in_range(self):
        for s in load_signal_library():
            for axis_id, effect in s.axis_effects.items():
                self.assertGreaterEqual(effect.level, 0.0)
                self.assertLessEqual(effect.level, 1.0)
                self.assertGreaterEqual(effect.importance, 0.0)
                self.assertLessEqual(effect.importance, 1.0)

    def test_ids_are_unique(self):
        ids = [s.id for s in load_signal_library()]
        self.assertEqual(len(ids), len(set(ids)))


class TestSignalsToAxes(unittest.TestCase):

    def setUp(self):
        self.library = load_signal_library()

    def test_no_signals_gives_empty_axes(self):
        self.assertEqual(signals_to_axes([], self.library), {})

    def test_single_signal_produces_its_axis_effects(self):
        detected = [DetectedSignal(signal_id="cadre_reglementaire", source_phrase="environnement réglementé")]
        result = signals_to_axes(detected, self.library)
        self.assertIn("cognitive_structuring", result)
        self.assertEqual(result["cognitive_structuring"]["level"], 0.8)

    def test_unknown_signal_id_is_ignored_not_crashing(self):
        detected = [DetectedSignal(signal_id="does_not_exist", source_phrase="n/a")]
        self.assertEqual(signals_to_axes(detected, self.library), {})

    def test_two_signals_same_axis_take_the_max_not_average(self):
        # coordination_multi_equipes -> social_interaction 0.85
        # any other signal touching social_interaction at a lower level must
        # NOT dilute the result via averaging.
        detected = [
            DetectedSignal(signal_id="coordination_multi_equipes", source_phrase="coordination avec plusieurs équipes"),
        ]
        result = signals_to_axes(detected, self.library)
        expected = 0.85
        self.assertEqual(result["social_interaction"]["level"], expected)

        # Duplicate the same signal (as if detected twice) — result must stay
        # identical, not double-count.
        detected_twice = detected + detected
        result_twice = signals_to_axes(detected_twice, self.library)
        self.assertEqual(result_twice["social_interaction"]["level"], expected)

    def test_deterministic_same_input_same_output(self):
        detected = [
            DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x"),
            DetectedSignal(signal_id="innovation_produit", source_phrase="y"),
        ]
        r1 = signals_to_axes(detected, self.library)
        r2 = signals_to_axes(detected, self.library)
        self.assertEqual(r1, r2)

    def test_multiple_signals_combine_across_different_axes(self):
        detected = [
            DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x"),
            DetectedSignal(signal_id="innovation_produit", source_phrase="y"),
        ]
        result = signals_to_axes(detected, self.library)
        self.assertIn("cognitive_structuring", result)   # from cadre_reglementaire
        self.assertIn("exploration", result)              # from innovation_produit


class TestCitationVerification(unittest.TestCase):
    """extract_signals() must not trust the LLM's source_phrase on its word —
    it's asked to quote verbatim, but only a programmatic substring check
    actually confirms it did."""

    def setUp(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake-key-for-test"
        self.library = load_signal_library()
        self.description = "Vous pilotez des projets innovants dans un environnement réglementé."

    def _mock_response(self, signals_json: str):
        resp = MagicMock()
        resp.content = [MagicMock(text=signals_json)]
        return resp

    def test_genuine_verbatim_citation_is_kept(self):
        payload = json.dumps({"signals": [
            {"signal_id": "innovation_produit", "source_phrase": "projets innovants"},
        ]})
        with patch("engine.job_extraction.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = self._mock_response(payload)
            detected = extract_signals(self.description, self.library)
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].signal_id, "innovation_produit")

    def test_fabricated_citation_not_in_source_text_is_dropped(self):
        payload = json.dumps({"signals": [
            {"signal_id": "innovation_produit", "source_phrase": "une phrase qui n'existe pas dans le texte"},
        ]})
        with patch("engine.job_extraction.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = self._mock_response(payload)
            detected = extract_signals(self.description, self.library)
        self.assertEqual(detected, [])

    def test_citation_with_curly_quotes_and_linewrap_still_verifies(self):
        # Same normalization discipline as EvidenceAble's citation checks:
        # a genuine match shouldn't fail on typographic quote style or a
        # hyphenation line-wrap artifact.
        description = "L'environnement est très r\u00e9glement\u00e9 et l'\u00e9quipe pilote des projets innovants."
        payload = json.dumps({"signals": [
            {"signal_id": "cadre_reglementaire", "source_phrase": "tr\u00e8s r\u00e9glement\u00e9"},
        ]})
        with patch("engine.job_extraction.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = self._mock_response(payload)
            detected = extract_signals(description, self.library)
        self.assertEqual(len(detected), 1)

    def test_mixed_genuine_and_fabricated_keeps_only_genuine(self):
        payload = json.dumps({"signals": [
            {"signal_id": "innovation_produit", "source_phrase": "projets innovants"},
            {"signal_id": "cadre_reglementaire", "source_phrase": "texte totalement inventé"},
        ]})
        with patch("engine.job_extraction.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = self._mock_response(payload)
            detected = extract_signals(self.description, self.library)
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].signal_id, "innovation_produit")


if __name__ == "__main__":
    unittest.main()
