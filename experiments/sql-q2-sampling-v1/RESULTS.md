# Q2 sampling (A1500, 2 attempts per question, temperature 0.8): all three parts PASS (2026-11-09/10)

Approved and launched once each from `babe5b8` on hf-user (amendments 11-13). Every registered check
passes for every part:
- completed with exit code 0;
- account and registration SHA match;
- every Q2 reference re-checked in the job before model use (SQLite 3.40.1), with 0 model calls;
- exact reload of the A1500 step-180 adapter (`69f0f125…`, 504 tensors);
- one finite projected position;
- AuditKIT agrees with every score;
- no OOM.

| Part | Job | Questions | Attempts | Strict successes | Running | Cost (rate) | GPU peak |
|---|---|---:|---:|---:|---|---:|---:|
| p1 | 6af179e3e6320c66fad806c5 | 193 | 386 | 112 | 8.43 h | $3.37 | 4.96 GiB |
| p2 | 6af17a120e0338d1bb8548d0 | 193 | 386 | 133 | 7.30 h | $2.92 | 5.06 GiB |
| p3 | 6af17a39e6320c66fad80749 | 191 | 382 | 135 | 7.05 h | $2.82 | 4.76 GiB |
| **Total** | | **577** | **1,154** | **380 (32.9%)** | 22.8 GPU-h | **$9.11** | |

The two tie-dependent Q2 questions were excluded before sampling (amendment 12).

## Self-training build (`scripts/build_q2_selftrain.py`, local, no model)

The B / A+ data is in `data/sqale/selftrain-v1/`, with the summary in `summary.json`.

- **Filter:** 380 attempts are eligible (clean protocol and guarded strict success), on 231 of 577
  questions, so **G2 is 40%, against a 25% bar: pass**.
- **Drops:** 71 duplicates or over the two-per-question cap. 3 more were dropped because their tool
  observations differ under this machine's SQLite 3.53.1.
- **Kept:** 306 verified episodes on 229 questions. Only **3** contain a recovery (a failed call followed
  by success).
- **Size gate:** 49,717 unique B targets per pass, 15.0% of A1500's 330,742, against a 10% bar: **pass**.
- **Control:** A+ gold-path episodes on the same 229 questions are matched to exactly 49,717 targets.

Most eligible questions are simple (127 one-table lookups, 66 two-table, 12 aggregates). Only 26 are
complex (10 with 3+ tables, 16 multi-step).
