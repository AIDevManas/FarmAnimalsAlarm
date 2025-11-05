Project Structure

This document describes the new professional layout added to the repository.

- `src/` - Python source package(s).
  - `farm_animals_alarm/` - package placeholder for library code.
- `app/` - existing GUI and application scripts (kept in place to preserve history).
- `models/` - store model weights here (add to .gitignore to avoid large commits).
- `data/` - any small sample data used by tests or demos.
- `notebooks/` - exploratory work and demos.
- `scripts/` - convenience scripts to run or debug the application.
- `tests/` - pytest tests.
- `docs/` - documentation and notes.

Recommendations:
- Move reusable logic out of `app/` into `src/farm_animals_alarm/` in future refactors.
- Keep model artifacts in `models/` and use a download script if needed.
- Add CI (GitHub Actions) to run tests automatically.
