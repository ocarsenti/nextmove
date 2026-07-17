"""Research persistence and reliability analysis for the NextMove validation study.

Deliberately separate from the product's runtime (no persistence there) and
from any future product database — a dedicated SQLite file, participant
codes instead of accounts, and only raw answers stored (see storage.py).
"""
