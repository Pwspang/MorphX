#!/bin/bash
# Put the shards of scripts/ae/morphx-nscc2b-shard.pbs together as a normal
# ./run.sh run directory (<RUN>/part1-colocate, ...) and regenerate tables/
# and figures/ from it with ./run.sh analyse.
#   bash scripts/ae/merge_nscc2b_shards.sh ~/scratch/morphx/MorphX/logs/20261026-nscc2b
set -euo pipefail
RUN=$(cd "$1" && pwd)
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
module load cuda/13.0 >/dev/null 2>&1 || true
source "${MORPHX_VENV:-$HOME/scratch/morphx/envs/sglang-stock}/bin/activate"
source scripts/ae/env-nscc2b.sh
# Must match the shards: point names include the scaled rates.
export MORPHX_EXTRA_SCHEDULERS="${MORPHX_EXTRA_SCHEDULERS-all}"
export MORPHX_GGNN_RPS_SCALE="${MORPHX_GGNN_RPS_SCALE:-2}"

part_of() {
  case "$1" in
    p1-*) echo part1-colocate ;;
    p2-*) echo part2-pd-colocate ;;
    p3) echo part3-overhead ;;
    p4) echo part4-model ;;
  esac
}
for shard in "$RUN"/shards/*/; do
  name=$(basename "$shard")
  part=$(part_of "$name")
  [[ -n "$part" ]] || { echo "skip unknown shard $name"; continue; }
  mkdir -p "$RUN/$part"
  # Per-run logs (and part 2's per-config files) never collide between shards;
  # the per-shard manifest/commands/tables are regenerated or informational.
  rsync -a --exclude tables --exclude mps --exclude runtime-libs "$shard" "$RUN/$part/"
  echo "merged $name -> $part"
done

[[ -f "$RUN/manifest.txt" ]] || printf 'created_at=%s\nrepo_root=%s\nmode=full\nsource=nscc2b shards\n' \
  "$(date -Iseconds)" "$PWD" > "$RUN/manifest.txt"
mkdir -p logs
ln -sfn "$RUN" logs/latest
./run.sh analyse
