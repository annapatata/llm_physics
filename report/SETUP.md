# Building the Report

## Prerequisites

You need a LaTeX distribution with `pdflatex` and `bibtex`. No extra style files are required.

### Windows

Install [MiKTeX](https://miktex.org/download). It auto-installs missing packages on first compile.

Verify:
```powershell
pdflatex --version
bibtex --version
```

### Linux

Install TeX Live:
```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

Verify:
```bash
pdflatex --version
bibtex --version
```

---

## Compile

### Windows (PowerShell)

```powershell
cd report
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

### Linux or Windows (Git Bash)

```bash
bash report/compile.sh
```

Output: `report/report.pdf`

---

## Open the PDF

**Windows:**
```powershell
start report\report.pdf
```

**Linux:**
```bash
xdg-open report/report.pdf
```

---

## File overview

| File | Purpose |
|------|---------|
| `report.tex` | Main LaTeX source |
| `references.bib` | BibTeX bibliography |
| `compile.sh` | Compile script (Linux / Git Bash) |
