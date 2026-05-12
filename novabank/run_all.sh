#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -m src.data
python -m src.models
python -m src.evaluate
python -m src.decision
python -m src.explain
cd reports
pandoc exec_memo.md -o exec_memo.html --standalone -c memo_style.css --metadata title="NovaBank Executive Memo"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer --no-margins \
    --print-to-pdf="$PWD/exec_memo.pdf" "file://$PWD/exec_memo.html"
fi
cd ..
echo "Done. Artifacts in reports/"
