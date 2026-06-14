# Building the Report

## Prerequisites

Install [MiKTeX](https://miktex.org/download) (Windows). It will auto-install any missing LaTeX packages on first compile.

Verify your installation:
```powershell
pdflatex --version
bibtex --version
```

---

## Compile (PowerShell)

```powershell
cd report
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

Output: `report/report.pdf`

You need to run `pdflatex` three times — the first pass builds the structure, `bibtex` resolves citations, and the final two passes fill in cross-references and the bibliography.

---

## Open the PDF

```powershell
start report\report.pdf
```

---

## File overview

| File | Purpose |
|------|---------|
| `report.tex` | Main LaTeX source |
| `references.bib` | BibTeX bibliography |
| `compile.sh` | Compile script for Git Bash |
