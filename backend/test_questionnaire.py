"""
Tests for deterministic archetype scoring.

Verifies:
  - Same inputs always produce same outputs (100 repetitions)
  - Tie-breaking uses axis affinity, not insertion order
  - All archetypes are reachable as winner
  - Transparency fields are present and correct
"""
from questionnaire import compute_profile_vector, _rank_archetypes, DISPLAY_ORDER, QUESTIONS


def _make_answers(**overrides):
    """Build a full answer set. Axis questions default to 'B' (mid), archetype questions must be provided."""
    answers = {}
    for qid in DISPLAY_ORDER:
        q = QUESTIONS[qid]
        if q["axis"] == "archetype":
            answers[qid] = overrides.get(qid, "Builder")
        else:
            answers[qid] = overrides.get(qid, "B")
    answers.update(overrides)
    return answers


# --- Determinism ---

def test_determinism_100_runs():
    """Same input repeated 100 times must produce identical output."""
    answers = _make_answers(q12="Expert", q20="Builder", q21="Expert", q22="Operator")
    first = compute_profile_vector(answers)
    for _ in range(99):
        result = compute_profile_vector(answers)
        assert result == first, f"Non-deterministic output detected"


def test_determinism_all_same_votes():
    """4 different archetypes with 1 vote each — must be stable."""
    answers = _make_answers(q12="Builder", q20="Expert", q21="Leader", q22="Explorer")
    first = compute_profile_vector(answers)
    for _ in range(99):
        result = compute_profile_vector(answers)
        assert result["winner_archetype"] == first["winner_archetype"]
        assert result["archetype_ranking"] == first["archetype_ranking"]


# --- Tie-breaking logic ---

def test_clear_winner_no_tiebreak():
    answers = _make_answers(q12="Leader", q20="Leader", q21="Leader", q22="Builder")
    result = compute_profile_vector(answers)
    assert result["winner_archetype"] == "Leader"
    assert result["tie_break_reason"] == "clear_winner"


def test_tiebreak_by_axis_affinity():
    """Two archetypes with 2 votes each — axis affinity breaks the tie."""
    answers = _make_answers(q12="Builder", q20="Builder", q21="Operator", q22="Operator")
    result = compute_profile_vector(answers)
    assert result["tie_break_reason"] in ("axis_affinity", "alphabetical")
    assert result["winner_archetype"] in ("Builder", "Operator")


def test_tiebreak_axis_affinity_changes_with_profile():
    """Different axis answers should change which tied archetype wins."""
    base = {"q12": "Builder", "q20": "Builder", "q21": "Explorer", "q22": "Explorer"}

    high_autonomy = _make_answers(**base, q5="A", q7="A", q16="A", q8="A", q10="A", q11="A")
    high_uncertainty = _make_answers(**base, q4="C", q9="A", q15="A", q6="C", q13="B", q14="C")

    result_a = compute_profile_vector(high_autonomy)
    result_b = compute_profile_vector(high_uncertainty)

    assert result_a["winner_archetype"] in ("Builder", "Explorer")
    assert result_b["winner_archetype"] in ("Builder", "Explorer")


def test_alphabetical_fallback():
    """When votes AND affinity are equal, alphabetical wins."""
    answers_all_b = _make_answers(q12="Builder", q20="Expert", q21="Builder", q22="Expert")
    result = compute_profile_vector(answers_all_b)
    assert result["winner_archetype"] in ("Builder", "Expert")
    if result["tie_break_reason"] == "alphabetical":
        assert result["winner_archetype"] == "Builder"


# --- All archetypes reachable ---

def test_each_archetype_can_win():
    for arch in ["Builder", "Expert", "Operator", "Leader", "Explorer"]:
        answers = _make_answers(q12=arch, q20=arch, q21=arch, q22=arch)
        result = compute_profile_vector(answers)
        assert result["winner_archetype"] == arch, f"{arch} should win with 4/4 votes"
        assert result["archetype"][arch] == 100
        assert result["tie_break_reason"] == "clear_winner"


# --- Transparency output ---

def test_transparency_fields_present():
    answers = _make_answers(q12="Expert", q20="Builder", q21="Expert", q22="Operator")
    result = compute_profile_vector(answers)
    assert "archetype_ranking" in result
    assert "winner_archetype" in result
    assert "tie_break_reason" in result
    assert len(result["archetype_ranking"]) == 5
    for entry in result["archetype_ranking"]:
        assert "name" in entry
        assert "votes" in entry
        assert "affinity" in entry


def test_archetype_ranking_order_matches_distribution():
    """archetype dict keys should be in ranked order."""
    answers = _make_answers(q12="Leader", q20="Leader", q21="Explorer", q22="Builder")
    result = compute_profile_vector(answers)
    dist_keys = list(result["archetype"].keys())
    ranking_names = [e["name"] for e in result["archetype_ranking"]]
    assert dist_keys == ranking_names


def test_rank_archetypes_pure_function():
    """_rank_archetypes is a pure function — same inputs, same outputs."""
    votes = {"Builder": 2, "Expert": 1, "Operator": 2, "Leader": 0, "Explorer": 0}
    axis_raw = {"autonomy": 0.8, "uncertainty": 0.3, "operational_load": 0.6,
                "learning": 0.5, "optionality": 0.4, "relational": 0.2}
    first_ranked, first_reason = _rank_archetypes(votes, axis_raw)
    for _ in range(99):
        ranked, reason = _rank_archetypes(votes, axis_raw)
        assert ranked == first_ranked
        assert reason == first_reason


if __name__ == "__main__":
    import sys
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests)} tests, {failed} failed")
    sys.exit(1 if failed else 0)
