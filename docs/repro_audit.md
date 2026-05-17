# VN2 reproducibility audit

This branch adds a manual GitHub Actions workflow for checking whether the VN2 solution is executable and competition-compliant from a clean CI runner.

## Workflow

File: `.github/workflows/vn2-audit.yml`

Trigger manually from GitHub Actions with `workflow_dispatch`.

Modes:

- `smoke`: installs dependencies, audits core competition mechanics, and validates committed submissions if present.
- `full`: does everything in `smoke`, then runs the canonical final policy from `best_winner.json`, writes six submission CSVs, validates them, and uploads logs plus CSV artifacts.

## Canonical final entrypoint

```bash
python scripts/run_final_audit.py --write-submissions --submission-dir submissions
python scripts/validate_submissions.py --submission-dir submissions
```

The final policy is loaded from `best_winner.json`, not duplicated by hand in the workflow. This makes the audit result traceable to the documented policy config.

## What the audit checks

- Competition constants: holding cost, shortage cost, 2-week lead time, 6 rounds, 8-week horizon.
- Required data files and submission template shape.
- Simulator scoring split: setup weeks 1-2 vs competition weeks 3-8.
- Six submission files, 599 rows each, canonical Store/Product row order, no missing values, non-negative integer quantities.
- Full-mode artifact hashes for generated `submissions/round_*.csv`.

## Interpreting results

A passing `smoke` run means the repository mechanics and committed submissions are structurally valid.

A passing `full` run means GitHub Actions can reproduce the canonical final policy execution and produce valid submission files on the checked-out commit. Compare `audit_artifacts/final_results.json` against `best_winner.json` to confirm the reported sacred cost matches the documented seed-42 result.
