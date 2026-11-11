"""Scoring-side SQL limits, separate from the agent's tool limits (amendments 11 and 12).

Scoring re-executes reference (gold) queries and the agent's submitted/fallback queries. The agent tools
(sqlenv) keep sqlscore's defaults: 5 s and 50M SQLite operations.

- Amendment 11: on large databases (BIRD) a reference taking ~3 s locally can exceed 5 s on a slower or
  busy machine, which aborted runs or marked references ineligible, so scoring got its own time limit.
- Amendment 12: the job image runs SQLite 3.40.1 while the data was prepared on 3.53.1. Under 3.40 three
  BIRD references need 334M-1,193M operations (up to ~18 s locally) instead of 1.6M-12.5M, so the first
  final-BIRD job stopped at its reference check. Scoring now allows 2,000M operations and 60 s.
"""
SCORING_SECONDS = 60.0
SCORING_MAX_OPS = 2_000_000_000
