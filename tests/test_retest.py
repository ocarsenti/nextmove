"""Tests for the test-retest reproducibility study: store, analysis, API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from engine.retest_store import RetestStore, UnknownParticipantCode
from engine.retest_analysis import compute_retest_report, _icc_1_1, _pearson
from api.main import app, _questionnaire

client = TestClient(app)


def _full_answers(value: str = "B") -> dict[str, str]:
    base = _questionnaire.initial_sequence()
    return {qid: value for qid in base}


def _passage_kwargs(**overrides) -> dict:
    kwargs = dict(
        session_user_id="user-1",
        context_id=None,
        registry_version=1,
        answers={"q1": "A"},
        axis_scores={"autonomy": {"raw_value": 0.6}, "pace": {"raw_value": 0.4}},
        weighted_scores={"autonomy": 0.6, "pace": 0.4},
        archetype_dominant="Builder",
        archetype_secondary="Expert",
        is_complete=True,
    )
    kwargs.update(overrides)
    return kwargs


# ===================================================================
# STORE
# ===================================================================

class TestRetestStore:
    def test_first_passage_generates_code(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        code, passage_number = store.save_passage(**_passage_kwargs())
        assert passage_number == 1
        assert "-" in code

    def test_second_passage_links_via_code(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        code, _ = store.save_passage(**_passage_kwargs())
        code2, passage_number = store.save_passage(
            **_passage_kwargs(participant_code=code)
        )
        assert code2 == code
        assert passage_number == 2

    def test_unknown_code_raises(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        with pytest.raises(UnknownParticipantCode):
            store.save_passage(**_passage_kwargs(participant_code="nope-nope"))

    def test_paired_first_two_only_counts_complete_pairs(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        code, _ = store.save_passage(**_passage_kwargs())
        # Incomplete passage elsewhere shouldn't produce a pair.
        store.save_passage(**_passage_kwargs(session_user_id="user-2", is_complete=False))
        assert store.paired_first_two() == []

        store.save_passage(**_passage_kwargs(participant_code=code))
        pairs = store.paired_first_two()
        assert len(pairs) == 1
        first, second = pairs[0]
        assert first["passage_number"] == 1
        assert second["passage_number"] == 2


# ===================================================================
# ANALYSIS
# ===================================================================

class TestRetestAnalysis:
    def test_perfect_agreement_gives_icc_and_r_of_one(self):
        xs = [0.2, 0.5, 0.8, 0.9, 0.3]
        assert _icc_1_1(xs, list(xs)) == pytest.approx(1.0)
        assert _pearson(xs, list(xs)) == pytest.approx(1.0)

    def test_systematic_shift_keeps_correlation_but_lowers_icc(self):
        xs = [0.2, 0.5, 0.8, 0.9, 0.3]
        shifted = [x + 0.2 for x in xs]
        assert _pearson(xs, shifted) == pytest.approx(1.0)
        assert _icc_1_1(xs, shifted) < 0.9

    def test_empty_store_returns_empty_report(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        report = compute_retest_report(store)
        assert report["n_pairs"] == 0
        assert report["sample_sufficient"] is False
        assert report["axes"] == {}

    def test_report_computes_per_axis_metrics(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        code, _ = store.save_passage(**_passage_kwargs(
            axis_scores={"autonomy": {"raw_value": 0.6}},
        ))
        store.save_passage(**_passage_kwargs(
            participant_code=code,
            axis_scores={"autonomy": {"raw_value": 0.65}},
        ))
        report = compute_retest_report(store, tolerance=0.10)
        assert report["n_pairs"] == 1
        axis = report["axes"]["autonomy"]
        assert axis["n"] == 1
        assert axis["mean_abs_delta"] == pytest.approx(0.05)
        assert axis["pct_within_tolerance"] == 1.0

    def test_sample_sufficient_flag(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        for i in range(20):
            code, _ = store.save_passage(**_passage_kwargs(session_user_id=f"user-{i}"))
            store.save_passage(**_passage_kwargs(
                session_user_id=f"user-{i}", participant_code=code,
            ))
        report = compute_retest_report(store)
        assert report["n_pairs"] == 20
        assert report["sample_sufficient"] is True


# ===================================================================
# API
# ===================================================================

@pytest.mark.api
class TestRetestEndpoints:
    def test_save_first_then_second_passage(self, monkeypatch, tmp_path):
        import api.main as main_module

        monkeypatch.setattr(main_module, "_retest_store", RetestStore(tmp_path / "retest.db"))

        answers = _full_answers()
        r1 = client.post("/study/retest/save", json={"user_id": "u1", "answers": answers})
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["is_new_participant"] is True
        assert body1["passage_number"] == 1

        r2 = client.post(
            "/study/retest/save",
            json={"user_id": "u1", "answers": answers, "participant_code": body1["participant_code"]},
        )
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["is_new_participant"] is False
        assert body2["passage_number"] == 2

    def test_save_unknown_code_returns_404(self, monkeypatch, tmp_path):
        import api.main as main_module

        monkeypatch.setattr(main_module, "_retest_store", RetestStore(tmp_path / "retest.db"))

        r = client.post(
            "/study/retest/save",
            json={"user_id": "u1", "answers": _full_answers(), "participant_code": "zzz-zzz"},
        )
        assert r.status_code == 404

    def test_report_endpoint_returns_shape(self, monkeypatch, tmp_path):
        import api.main as main_module

        monkeypatch.setattr(main_module, "_retest_store", RetestStore(tmp_path / "retest.db"))

        r = client.get("/study/retest/report")
        assert r.status_code == 200
        body = r.json()
        assert "n_pairs" in body
        assert "axes" in body
