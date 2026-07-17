"""Tests for the test-retest reproducibility study: store, analysis, API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from engine.retest_store import RetestStore, UnknownParticipantCode
from engine.retest_analysis import compute_retest_report, _icc_1_1, _pearson
from engine.quality_report import compute_quality_report
from api.main import app, _questionnaire, _bank, _registry

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


def _full_axis_score(axis_id: str, raw_value: float, variance: float = 0.02) -> dict:
    """A complete AxisScore-shaped dict — required for quality_report tests,
    which reconstruct real AxisScore/Profile objects (unlike the retest
    analysis tests above, which only read `raw_value` out of raw JSON)."""
    return {
        "axis_id": axis_id,
        "raw_value": raw_value,
        "confidence": 0.8,
        "n_questions": 3,
        "variance": variance,
        "raw_answers": [raw_value],
        "effective_weight": 1.0,
        "status": "active",
    }


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

    def test_all_complete_passages_counts_singles_and_pairs(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        store.save_passage(**_passage_kwargs(session_user_id="user-1"))  # single, no retest
        code, _ = store.save_passage(**_passage_kwargs(session_user_id="user-2"))
        store.save_passage(**_passage_kwargs(session_user_id="user-2", participant_code=code))
        store.save_passage(**_passage_kwargs(session_user_id="user-3", is_complete=False))

        rows = store.all_complete_passages()
        assert len(rows) == 3  # 1 single + 2 from the pair; incomplete one excluded


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
# QUALITY REPORT (internal consistency + inter-axis correlation)
# ===================================================================

class TestQualityReport:
    def test_empty_store_returns_empty_report(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        report = compute_quality_report(store, _bank, _registry)
        assert report["n_profiles"] == 0
        assert report["sample_sufficient"] is False
        assert report["axes"] == {}
        assert report["correlations"] == []

    def test_single_passage_counts_as_one_profile_no_pair_needed(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        store.save_passage(**_passage_kwargs(
            axis_scores={"autonomy": _full_axis_score("autonomy", 0.6)},
        ))
        report = compute_quality_report(store, _bank, _registry)
        assert report["n_profiles"] == 1

    def test_incomplete_passage_excluded(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        store.save_passage(**_passage_kwargs(
            axis_scores={"autonomy": _full_axis_score("autonomy", 0.6)},
            is_complete=False,
        ))
        report = compute_quality_report(store, _bank, _registry)
        assert report["n_profiles"] == 0

    def test_highly_correlated_axes_are_flagged(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        # autonomy and pace move in lockstep across profiles; rigor is constant
        # (SignalComputer needs n>=3 profiles and non-zero variance to report r).
        for i, (a, p) in enumerate([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]):
            store.save_passage(**_passage_kwargs(
                session_user_id=f"user-{i}",
                axis_scores={
                    "autonomy": _full_axis_score("autonomy", a),
                    "pace": _full_axis_score("pace", p),
                    "rigor": _full_axis_score("rigor", 0.5 + i * 0.2),
                },
            ))
        report = compute_quality_report(store, _bank, _registry)
        assert report["n_profiles"] == 3

        pair = {frozenset([c["axis_a"], c["axis_b"]]): c for c in report["correlations"]}
        autonomy_pace = pair[frozenset(["autonomy", "pace"])]
        assert autonomy_pace["correlation"] == pytest.approx(1.0, abs=1e-6)
        assert autonomy_pace["flagged_redundant"] is True

        # Every unordered pair appears exactly once, not twice (SignalSet
        # stores both (a, b) and (b, a) internally).
        assert len(report["correlations"]) == 3

    def test_sample_sufficient_flag(self, tmp_path):
        store = RetestStore(tmp_path / "retest.db")
        for i in range(30):
            store.save_passage(**_passage_kwargs(
                session_user_id=f"user-{i}",
                axis_scores={"autonomy": _full_axis_score("autonomy", 0.1 * (i % 10))},
            ))
        report = compute_quality_report(store, _bank, _registry)
        assert report["n_profiles"] == 30
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

    def test_quality_report_endpoint_returns_shape(self, monkeypatch, tmp_path):
        import api.main as main_module

        monkeypatch.setattr(main_module, "_retest_store", RetestStore(tmp_path / "retest.db"))

        r = client.get("/study/quality/report")
        assert r.status_code == 200
        body = r.json()
        assert "n_profiles" in body
        assert "axes" in body
        assert "correlations" in body
