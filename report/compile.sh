#!/usr/bin/env bash
# Compiles report.tex to PDF using pdflatex + bibtex
# Works on Linux (TeX Live) and Windows (MiKTeX via Git Bash)
set -e

cd "$(dirname "$0")"

echo "==> Pass 1: pdflatex"
pdflatex -interaction=nonstopmode report.tex

echo "==> bibtex"
bibtex report

echo "==> Pass 2: pdflatex"
pdflatex -interaction=nonstopmode report.tex

echo "==> Pass 3: pdflatex"
pdflatex -interaction=nonstopmode report.tex

echo "==> Cleaning build artefacts"
rm -f report.aux report.bbl report.blg report.log report.out report.toc

echo "==> Done. Output: report/report.pdf"
