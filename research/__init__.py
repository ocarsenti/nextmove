"""Research/validation tooling for NextMove — separate from the product's
own runtime state (which has no persistence layer of its own beyond the
candidate reliability study in engine/retest_store.py).

job_extraction_log.py / job_extraction_validation.py: validation of the
job-side signal extraction pipeline (engine/job_extraction.py) — a
distinct problem from candidate questionnaire reliability, since the
"instrument" here is an LLM reading free text, not a fixed set of items
answered by a person.
"""
