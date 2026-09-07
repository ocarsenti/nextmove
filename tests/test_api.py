"""API-level tests for NextMove V5/V6 (FastAPI TestClient).

These tests exercise the HTTP layer directly — request/response shapes,
status codes, and cross-endpoint flows (score -> jobs -> match) — which
the engine-level unit tests deliberately don't cover.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import app, _questionnaire

client = TestClient(app)


def _full_answers(value: str = "B") -> dict[str, str]:
    """All base questions answered with the same value — mirrors the
    helper used in tests/test_v5.py so both suites stay consistent."""
    base = _questionnaire.initial_sequence()
    return {qid: value for qid in base}


# ===================================================================
# AXES
# ===================================================================

@pytest.mark.api
class TestAxesEndpoints:
    def test_get_axes_returns_18(self):
        r = client.get("/axes")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 18
        assert len(body["axes"]) == 18

    def test_get_single_axis(self):
        axis_id = client.get("/axes").json()["axes"][0]["id"]
        r = client.get(f"/axes/{axis_id}")
        assert r.status_code == 200
        assert r.json()["id"] == axis_id

    def test_get_unknown_axis_404(self):
        r = client.get("/axes/does_not_exist")
        assert r.status_code == 404


# ===================================================================
# QUESTIONNAIRE
# ===================================================================

@pytest.mark.api
class TestQuestionnaireEndpoints:
    def test_get_questionnaire_non_empty(self):
        r = client.get("/questionnaire")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == body["base_count"]
        assert body["total"] > 0

    def test_next_questions_with_partial_answers(self):
        base = _questionnaire.initial_sequence()
        partial = {qid: "B" for qid in base[:3]}
        r = client.post("/questionnaire/next", json={
            "user_id": "u1", "answers": partial, "already_seen": list(partial.keys()),
        })
        assert r.status_code == 200
        body = r.json()
        assert body["remaining_base"] == len(base) - 3


# ===================================================================
# SCORE
# ===================================================================

@pytest.mark.api
class TestScoreEndpoint:
    def test_score_full_profile_is_complete(self):
        r = client.post("/score", json={
            "user_id": "u1", "answers": _full_answers(), "context_hint": None,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["is_complete"] is True
        assert body["user_id"] == "u1"
        assert "archetype" in body and body["archetype"]

    def test_score_partial_profile_reports_pending_questions(self):
        base = _questionnaire.initial_sequence()
        partial = {qid: "B" for qid in base[:2]}
        r = client.post("/score", json={"user_id": "u2", "answers": partial})
        assert r.status_code == 200
        body = r.json()
        assert body["is_complete"] is False
        assert len(body["pending_questions"]) > 0

    def test_score_with_startup_context_hint(self):
        r = client.post("/score", json={
            "user_id": "u3", "answers": _full_answers("A"), "context_hint": "startup",
        })
        assert r.status_code == 200
        assert r.json()["context_detected"] == "startup"


# ===================================================================
# JOB CARDS
# ===================================================================

@pytest.mark.api
class TestJobEndpoints:
    def _sample_job_payload(self, job_id="job_test_1"):
        axis_id = client.get("/axes").json()["axes"][0]["id"]
        return {
            "job_id": job_id,
            "title": "Test Job",
            "description": "A role used only for API tests.",
            "axis_requirements": {
                axis_id: {"level": 0.7, "importance": 0.8, "note": "test"}
            },
        }

    def test_create_and_fetch_job(self):
        payload = self._sample_job_payload()
        r = client.post("/jobs", json=payload)
        assert r.status_code == 200
        assert r.json()["version"] == 1

        r2 = client.get(f"/jobs/{payload['job_id']}")
        assert r2.status_code == 200
        assert r2.json()["title"] == "Test Job"

    def test_create_job_unknown_axis_rejected(self):
        payload = self._sample_job_payload(job_id="job_test_bad_axis")
        payload["axis_requirements"] = {"not_a_real_axis": {"level": 0.5, "importance": 0.5, "note": ""}}
        r = client.post("/jobs", json=payload)
        assert r.status_code == 400

    def test_list_jobs_includes_created_job(self):
        payload = self._sample_job_payload(job_id="job_test_list")
        client.post("/jobs", json=payload)
        r = client.get("/jobs")
        assert r.status_code == 200
        ids = [j["job_id"] for j in r.json()]
        assert "job_test_list" in ids

    def test_get_unknown_job_404(self):
        r = client.get("/jobs/does_not_exist")
        assert r.status_code == 404

    def test_job_constraints_for_unknown_job_404(self):
        r = client.get("/jobs/does_not_exist/constraints")
        assert r.status_code == 404


# ===================================================================
# MATCH — cross-endpoint flow: create job -> score candidate -> match
# ===================================================================

@pytest.mark.api
@pytest.mark.integration
class TestMatchFlow:
    def test_match_unknown_job_404(self):
        r = client.post("/match", json={
            "user_id": "u1", "answers": _full_answers(), "job_id": "no_such_job",
        })
        assert r.status_code == 404

    def test_full_match_flow(self):
        axis_id = client.get("/axes").json()["axes"][0]["id"]
        job_payload = {
            "job_id": "job_match_flow",
            "title": "Flow Test Job",
            "description": "",
            "axis_requirements": {axis_id: {"level": 0.6, "importance": 0.9, "note": ""}},
        }
        assert client.post("/jobs", json=job_payload).status_code == 200

        r = client.post("/match", json={
            "user_id": "u_flow", "answers": _full_answers(), "job_id": "job_match_flow",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == "job_match_flow"
        assert 0.0 <= body["fit_score"] <= 1.0
        assert "job_constraints" in body


# ===================================================================
# RULES
# ===================================================================

@pytest.mark.api
class TestRulesEndpoints:
    def test_get_rules_no_context(self):
        r = client.get("/rules")
        assert r.status_code == 200
        assert "active_rules" in r.json()

    def test_get_v6_rules(self):
        r = client.get("/v6/rules")
        assert r.status_code == 200
        body = r.json()
        assert body["total_rules"] > 0

    def test_evaluate_v6_rules_with_signals(self):
        r = client.post("/v6/rules/evaluate", json={"signals": {}, "context": None})
        assert r.status_code == 200
        assert "firings" in r.json()


# ===================================================================
# JOB EXTRACTION — LLM call is mocked, never hits the real API
# ===================================================================

@pytest.mark.api
class TestJobExtractEndpoint:
    def test_extract_without_api_key_returns_503(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        r = client.post("/jobs/extract", json={"description": "Some job text."})
        assert r.status_code == 503

    def test_extract_with_mocked_llm_response(self, monkeypatch):
        from engine import job_extraction

        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")

        library = job_extraction.load_signal_library()
        first_signal = library[0]

        class _FakeContentBlock:
            text = (
                '{"signals": [{"signal_id": "%s", "source_phrase": "Any description"}]}'
                % first_signal.id
            )

        class _FakeResponse:
            content = [_FakeContentBlock()]

        monkeypatch.setattr(
            job_extraction.Anthropic, "__init__", lambda self, api_key=None: None
        )
        monkeypatch.setattr(
            job_extraction.Anthropic,
            "messages",
            property(lambda self: type(
                "M", (), {"create": staticmethod(lambda **kw: _FakeResponse())}
            )()),
            raising=False,
        )

        r = client.post("/jobs/extract", json={"description": "Any description."})
        assert r.status_code == 200
        body = r.json()
        assert any(s["signal_id"] == first_signal.id for s in body["signals"])


# ===================================================================
# ARCHETYPE CALIBRATION LOGGING — /score should log raw_affinity for
# complete profiles only (see research/archetype_calibration_log.py)
# ===================================================================

@pytest.mark.api
class TestArchetypeCalibrationLogging:
    def test_complete_profile_is_logged(self, monkeypatch, tmp_path):
        from research import archetype_calibration_log as cal_log

        db_path = tmp_path / "calib_complete.db"
        cal_log.init_db(db_path)
        monkeypatch.setattr(cal_log, "DB_PATH", db_path)

        before = cal_log.count_passages(db_path)
        r = client.post("/score", json={"user_id": "u_calib_1", "answers": _full_answers()})
        assert r.status_code == 200
        assert r.json()["is_complete"] is True
        after = cal_log.count_passages(db_path)
        assert after == before + 1

        row = cal_log.all_passages(db_path)[-1]
        assert set(row["raw_affinity"].keys()) == {"Builder", "Expert", "Operator", "Leader", "Connecteur"}
        # no PII / no linkage back to this session's user_id
        assert "user_id" not in row and "session_user_id" not in row

    def test_partial_profile_is_not_logged(self, monkeypatch, tmp_path):
        from research import archetype_calibration_log as cal_log

        db_path = tmp_path / "calib_partial.db"
        cal_log.init_db(db_path)
        monkeypatch.setattr(cal_log, "DB_PATH", db_path)

        base = _questionnaire.initial_sequence()
        partial = {qid: "B" for qid in base[:2]}
        before = cal_log.count_passages(db_path)
        r = client.post("/score", json={"user_id": "u_calib_2", "answers": partial})
        assert r.status_code == 200
        assert r.json()["is_complete"] is False
        assert cal_log.count_passages(db_path) == before
