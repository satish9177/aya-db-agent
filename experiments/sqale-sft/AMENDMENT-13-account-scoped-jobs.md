# Amendment 13: each registration names the HF account that runs it (2026-11-09)

Written before the jobs it covers. Amendments 11 and 12 are unchanged. **No upload, job or push is approved
by this amendment.**

## What changes

- **Allowed accounts:** `runconfig.ACCOUNTS` lists the HF accounts registered jobs may run on.
  - `load_registration` accepts a registration whose `expected_account` is in it.
  - `require_execution` requires the launcher's established account to equal the registration's own
    `expected_account`.
- **Token:** `scripts/sql_job.py` uses the active token if it belongs to the registration's account, and
  otherwise the locally stored token that does. Tokens are never printed.
- **Bucket:** a job writes only to its account's private bucket `<account>/aya-db-agent`. The output-sync
  check uses the registration's account, not a fixed name. Receipts and pulls stay under
  `runs/hf-sql/<account>/` and `remote/hf-sql/<account>/`.

## Checked before any spend

The token resolves to the registration's account and can read the gated Tiny Aya Global revision
`af89d219…`. The free preflight checks the account, gated-weight access and the hardware price before any
upload.

## Unchanged

Same GPU type (t4-small), image, code, data, scoring limits (amendment 12) and frozen selection. Nothing
else changes.
