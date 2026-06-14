# Report Setup

## Dependencies

### 1. LaTeX distribution

Install [MiKTeX](https://miktex.org/download) (Windows). During first compile it will auto-install any missing packages.

Verify it is installed:
```bash
pdflatex --version
bibtex --version
```

### 2. ICML 2024 style file

Run the fetch script to download `icml2024.sty`:
```bash
bash report/fetch_template.sh
```

This downloads `icml2024.sty` directly from the ICML website and places it next to `report.tex`.

---

## Compiling

```bash
bash report/compile.sh
```

This runs `pdflatex → bibtex → pdflatex → pdflatex` and produces `report/report.pdf`.

---

## File overview

| File | Purpose |
|------|---------|
| `report.tex` | Main LaTeX source |
| `references.bib` | BibTeX bibliography |
| `fetch_template.sh` | Downloads `icml2024.sty` |
| `compile.sh` | Compiles the PDF |
