#!/bin/bash
# Poll AI-mode run 45032903620 (Nano ch338) every 5 min, up to ~70 min.
RUN=45032903620
REPO=tuantran00541-spec/manga-translator
for i in $(seq 1 14); do
  OUT=$(gh run view $RUN --repo $REPO --json status,conclusion --jq '{status, conclusion}' 2>&1)
  echo "[$(date '+%H:%M:%S')] poll $i: $OUT"
  ST=$(echo "$OUT" | grep -o '"status": "[^"]*"' | cut -d'"' -f4)
  if [ "$ST" = "completed" ]; then
    echo "=== RUN CONCLUDED ==="
    gh run view $RUN --repo $REPO --json status,conclusion,headSha --jq '{status, conclusion, headSha}'
    gh run view $RUN --repo $REPO --json jobs --jq '.jobs[] | "\(.name): \(.conclusion // .status)"'
    echo "=== ARTIFACTS ==="
    gh api repos/$REPO/actions/runs/$RUN/artifacts --jq '.artifacts[] | "\(.name) \(.size_in_bytes) bytes expired=\(.expired)"' 2>&1
    exit 0
  fi
  sleep 300
done
echo "=== STILL RUNNING AFTER ~70 MIN ==="
gh run view $RUN --repo $REPO --json status,conclusion --jq '{status, conclusion}'
exit 2
