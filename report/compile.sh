#!/usr/bin/env bash
# Compiles report.tex to PDF using pdflatex + bibtex
set -e

cd "$(dirname "$0")"

if [ ! -f "icml2024.sty" ]; then
    echo "icml2024.sty not found. Run fetch_template.sh first."
    exit 1
fi

echo "==> Pass 1: pdflatex"
pdflatex -interaction=nonstopmode report.tex

echo "==> bibtex"
bibtex report

echo "==> Pass 2: pdflatex"
pdflatex -interaction=nonstopmode report.tex

echo "==> Pass 3: pdflatex"
pdflatex -interaction=nonstopmode report.tex

echo "==> Done. Output: report.pdf"
