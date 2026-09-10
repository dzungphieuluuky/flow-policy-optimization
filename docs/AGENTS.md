# AGENTS — docs

## Overview
User-facing navigation + architecture reference. 12 md files (refreshed post-LIBERO-purge, MuJoCo-only). All cross-link via `docs/README.md`.

## Structure
```
README.md            entry map + mental model
project-structure.md directory tree roles
architecture.md      mermaid flow Hann→rFFT→K5 → joint DCT → flow transformer
modules.md           jump table per src/mfpo file
getting-started.md   smoke commands (tiny matrix, train.py single-run, notebook)
config-flags.md      env flags / TINY / LEGACY_MUJOCO / matrix filters
notebooks.md         build/execute/self-containment (61/47 cells)
testing.md           263 test functions + wallclock placeholder note
troubleshooting.md   sys.path, train.py exit codes, AnalysisError fallback
wandb.md             default-on gate (MFPO_DISABLE_WANDB=1) + metric namespaces
kaggle-offline-blackwell.md Blackwell offline runbook (MEASURED GPU content — surgical edits only)
```

## Where to Look
- `README.md:1` mental model diagram — start here for newcomers
- `architecture.md` mermaid — copy into papers
- `modules.md` table — fastest jump to `networks.py:179` / `dct_utils.py:223`
- `config-flags.md` — `MFPO_LEGACY_MUJOCO`, `MFPO_TINY_MATRIX`, `IS_KAGGLE` branching

## Conventions
- Keep 12-file set stable — one topic per file, no new pinned deps docs
- Link `src/mfpo/*.py:line` explicitly — line numbers drift, mention approx
- Do not duplicate `AGENTS.md` content verbatim — docs are prose, AGENTS is terse map

## Anti-Patterns
- DO NOT add tutorial/how-to beyond `getting-started.md` + `troubleshooting.md`
- DO NOT change `getting-started.md` smoke commands without testing `scripts/train.py`

## Commands
```bash
ls docs/
mkdocs  # not used — plain md only
```

## Notes
- Generated via delegate + manual patch; keep telegraphic, no fluff
