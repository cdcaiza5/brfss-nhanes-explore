# Original (stale) version of the IEEE paper

This folder preserves the **stale** version of the paper that was uploaded at the start of the conversation (the `.tex` source you sent in the first message). It is kept here for audit only.

## What is in here

- `original.tex` — verbatim copy of the source from the start of the conversation, with the only change being the addition of `\usepackage{url}` (required for the `\url{}` command used in `Principal_Data.bib`).
- `original.pdf` — the PDF produced by `pdflatex` from `original.tex`. 4 pages.
- `Principal_Data.bib` — copy of the bibliography, required to build the PDF.
- `Imagenes/` — copy of the figure folder, required for the two confusion-matrix figures.

## How to rebuild

```bash
cd paper/original
pdflatex original.tex
bibtex original
pdflatex original.tex
pdflatex original.tex
```

## What is stale about this version

The original paper makes several claims that the project has since revised. The main drifts:

- The 4-tier table (Leakage / Strong / Moderate / Weak) was removed from the canonical paper.
- The `_MENT14D` and `_MENTHLTH` rows are labelled "Leakage" in the original; the current paper keeps `MENTHLTH` as a predictor and only excludes `_MENT14D` (as a target near-duplicate).
- The `ace_score` feature is described as engineered; the current paper excludes it (all-missing source).
- The hyperparameters are described as "meticulously configured"; the current paper drops that wording and uses the actual library defaults + a hand-picked stable set, with a tuning loop tried and discarded.
- The Table II results (0.5-threshold metrics, 6 columns) are an old run; the current paper uses a 5-fold full-data F1-tuned run with 10 columns.

The current authoritative version is at `../main.tex`.
