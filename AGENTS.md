# Repository Guidelines

## Project Structure & Module Organization
- `src_code/`: Jupyter notebooks and Python utilities that reproduce results.
- `src_dev/`: Julia reimplementation of optimal control and RL code.
- `src_latex/`: Manuscript sources (`main.tex`, `sections/`, `references.bib`).
- `data/`: Raw datasets used by notebooks and figures.
- `visual_elements/`: Figure scripts, generated figures, and videos.
- `Project.toml`/`Manifest.toml`: Julia environment; `requirements.txt`/`environment.yml`: Python environment.

## Build, Test, and Development Commands
- Python setup (from repo root):
  - `python -m venv .quctrl` then activate (`.quctrl\Scripts\activate` on Windows).
  - `python -m pip install -r requirements.txt`.
- Julia setup:
  - `julia --project=. -e "using Pkg; Pkg.instantiate()"`.
- Run a Julia script (example):
  - `julia --project=. src_dev/reinforcement_learning/PG_state_prep.jl`.
- Regenerate figures:
  - `cd visual_elements/figs` then `python fig3.py` or `sh generate_all_figs.sh`.

## Coding Style & Naming Conventions
- Julia: `snake_case` for functions/variables, `UpperCamelCase` for types; mutating functions may end in `!`.
- Python: follow PEP 8 conventions and keep functions small and readable.
- Indentation: 4 spaces, no tabs. Prefer clear names over abbreviations.

## Testing Guidelines
- No dedicated automated test suite in this repo.
- Validate changes by running the relevant notebook/script and checking outputs (data files or plots).
- For quick sanity checks, run short-episode training calls in the Julia RL scripts.

## Commit & Pull Request Guidelines
- Commit messages follow a simple type prefix: `feat:`, `refactor:`, `minor:` (per recent history).
- Keep commits focused and describe the intent.
- PRs should include: a short summary, reproduction commands, and note any regenerated data/figures.

## Data & Artifacts
- Store generated figures in `visual_elements/figs/generated_figs/`.
- Keep raw data under `data/raw/` organized by figure.
- Avoid committing large derived artifacts unless they are required for reproducibility.
