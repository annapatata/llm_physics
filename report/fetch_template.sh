#!/usr/bin/env bash
# Downloads icml2024.sty into this directory
set -e

DEST="$(dirname "$0")/icml2024.sty"
URL="https://icml.cc/media/icml-2024/Styles/icml2024.sty"

if [ -f "$DEST" ]; then
    echo "icml2024.sty already exists, skipping."
else
    echo "Downloading icml2024.sty..."
    curl -L "$URL" -o "$DEST"
    echo "Saved to $DEST"
fi
