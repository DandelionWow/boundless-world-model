#!/bin/bash
set -euo pipefail

cat >&2 <<'EOF'
scripts/train.sh is deprecated because it confused model-only initialization
with full-state resume. Use one of the guarded entry points instead:

  Single GPU:  bash scripts/train_example.sh
  Multi GPU:   bash scripts/train_distributed_example.sh

Set TRAIN_MODE=fresh with INIT_CKPT, or TRAIN_MODE=resume with RESUME_FROM.
See docs/robotwin_bwm_training.md for complete commands.
EOF
exit 2
