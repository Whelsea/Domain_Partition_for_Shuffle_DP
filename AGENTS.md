# AGENTS.md — Domain Partition for Shuffle-DP

## Project Overview

Academic LaTeX paper targeting **CCS '26** (ACM SIGSAC Conference on Computer and Communications Security).
Topic: Domain Partitioning for Shuffle Differential Privacy — user-level shuffle-DP frameworks
for static, dynamic, and personalized settings.

**This is a pure LaTeX project. There is no application code, no tests, no CI/CD.**

## Repository Structure

```
CCS.tex                          # Main entry point (DO NOT rename)
main.tex                         # Deprecated entry point (fully commented out — ignore)
ref.bib                          # Bibliography (BibTeX, ~21 entries)
sec/                             # Paper sections (numbered for ordering)
  1-Introduction.tex             # WIP — mostly Chinese outline notes
  2-Preliminary.tex              # Notation, DP definitions, shuffle-DP model
  3-Naive Solutions.tex           # Challenges, objectives, naive approaches
  4-Static_user_shuffle_DP.tex   # Core framework — two-round & one-round protocols
  5-Dynamic_user_shuffle_DP.tex  # Dynamic setting extension
  6-Application.tex              # Freq estimation, range counting, median
  7-Extension.tex                # Personalized DP extension
  8-Experiment.tex               # Stub — not yet written
  9-Conclusion.tex               # Stub — not yet written
  Appendix.tex                   # Proofs (renumbers theorems from 4.x)
  Related_work.tex               # Empty — not yet written
  draft/                         # Earlier drafts (reference only, not compiled)
experiment/
  experiment.pdf                 # Pre-generated experiment results
  static/dynamic/                # Empty placeholder directory
```

## Build Commands

Entry point is `CCS.tex`. Standard pdflatex + bibtex pipeline:

```bash
# Full build (from project root)
pdflatex CCS.tex && bibtex CCS && pdflatex CCS.tex && pdflatex CCS.tex

# Quick rebuild (no bib changes)
pdflatex CCS.tex

# Clean build artifacts
rm -f CCS.aux CCS.bbl CCS.blg CCS.log CCS.out CCS.synctex.gz
```

There is no Makefile, no latexmk config, no CI pipeline. Build locally.

## Document Class & Packages

- **Class**: `\documentclass[sigconf, review]{acmart}` — ACM sigconf format, review mode
- **Bibliography**: `ACM-Reference-Format` style, `\bibliography{ref}`
- **Algorithms**: `algorithm2e` with `[ruled, linesnumbered]`, `\DontPrintSemicolon`
- **Graphics**: `graphicx`, `tikz` (with `mindmap, trees` libraries)
- **Other**: `enumitem`, `changepage`, `hyperref`, `xcolor`, `lipsum`

## Custom Commands (Annotation System)

| Command | Rendering | Usage |
|---------|-----------|-------|
| `\bbd{text}` | **Blue bold** | Emphasis for key terms |
| `\obd{text}` | **Orange bold [TODO: text]** | Action items — things to write/fix |
| `\rbd{text}` | **Red bold [QUESTION: text]** | Open questions needing discussion |
| `\com{text}` | Gray [COMMENT: text] | Reviewer/collaborator annotations |
| `\revone{text}` | Red text | Revision marker (reviewer 1) |
| `\revtwo{text}` | Orange text | Revision marker (reviewer 2) |
| `\revthr{text}` | Blue text | Revision marker (reviewer 3) |
| `\err` | Error (upright) | Shorthand for `\mathrm{Error}` |

When adding TODOs, use `\obd{...}`. When flagging questions, use `\rbd{...}`.

## Theorem Environments

All numbered by section. Defined in `CCS.tex`:

- `theorem`, `definition`, `proposition`, `lemma`, `corollary` — numbered `N.x`
- `observation`, `remark` — unnumbered (starred)
- `example` — numbered `N.x`

**Appendix note**: `Appendix.tex` resets theorem counter and uses `4.\arabic{theorem}` numbering.

## Label & Reference Conventions

### Label prefixes

| Type | Pattern | Example |
|------|---------|---------|
| Section | `sec:Name` | `\label{sec:Preliminary}` |
| Algorithm | `alg:description` | `\label{alg:Randomizer for Estimating tau}` |
| Theorem | `the:name` | `\label{the: static two-round}` |
| Lemma | `lem:name` | `\label{lem: Bit Counting Protocol}` |
| Definition | `def:name` | `\label{def: PSDP}` |

**Note**: Some labels have spaces after the colon (e.g., `the: static two-round`). Follow existing style.

### Cross-references

Always use `~` (non-breaking space) before `\ref`:
- `Section~\ref{sec:Notation}`
- `Algorithm~\ref{alg:dynamic_counting}`
- `Theorem~\ref{the: static two-round}`
- `Lemma~\ref{lem: Bit Counting Protocol}`

## Math Notation Conventions

Consistent notation used throughout — **do not deviate**:

| Symbol | Meaning |
|--------|---------|
| `n` | Number of users |
| `U`, `\mathcal{X} = [U]^d` | Domain size, input domain |
| `M` | Global max records per user |
| `m_i` | Records contributed by user `i` |
| `m_{\max}(D)` | Max contribution in dataset `D` |
| `m_\tau` | Estimated clipping threshold |
| `D`, `D_i` | Full dataset, user `i`'s dataset |
| `Q`, `\tilde{Q}` | True query, estimated query |
| `\varepsilon, \delta` | DP privacy parameters |
| `\beta` | Failure probability |
| `T` | Time horizon (dynamic setting) |
| `\mathcal{P}_Q`, `\mathcal{R}`, `\mathcal{S}`, `\mathcal{A}` | Protocol, Randomizer, Shuffler, Analyzer |
| `\mathbb{I}(\cdot)` | Indicator function |
| `\tilde{O}(\cdot)` | Big-O hiding polylog factors |

## Writing Style

- **Prose language**: English (formal academic)
- **Internal comments**: Chinese (中文) is used in `%` comments for author notes and outlines
- **Section headers**: `\section`, `\subsection`, `\paragraph` — no `\subsubsection`
- **Enumerations**: `enumerate` with custom labels `\item[(a)]`, `\item[(b)]` for conditions
- **Citations**: `\cite{key}` — BibTeX keys follow `authorYYYYkeyword` pattern (e.g., `dong2024almost`, `ghazi2021differentially`)
- **Algorithm style**: `algorithm2e` with `\KwIn`, `\KwParam`, `\For`, `\If`, `\eIf`, `\Return`, `\SetKw`, `\SetKwData`, `\SetKwFunction`

## File Naming

- Section files: `N-Title.tex` where N is section number, title uses underscores
- Filename with space exists: `3-Naive Solutions.tex` — handle carefully in `\input{}`
- BibTeX keys: `authorYYYYkeyword` (lowercase, no spaces)

## Key Editing Rules

1. **Never modify `main.tex`** — it is fully commented out and serves as archive only
2. **Never modify `CCS.bbl`** — it is auto-generated by bibtex
3. **Preserve all `% please check` comments** — these are author review markers
4. **Preserve all `\obd{}` and `\rbd{}` markers** — these track open items
5. **Use `\input{sec/N-Title}` without `.tex` extension** in CCS.tex (LaTeX convention)
6. **Keep `draft/` untouched** — historical reference only, not compiled
7. **Add new bibliography entries to `ref.bib`** — use consistent `authorYYYYkeyword` keys
8. **Equations**: Use `$$...$$` for display math (existing style), `$...$` for inline
9. **Footnotes**: Used sparingly for technical assumptions (see `4-Static_user_shuffle_DP.tex`)