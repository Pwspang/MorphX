#!/bin/bash
# Tables and figures from whichever nscc2b shards have finished so far, so each
# figure can be looked at as soon as its shard is done (the full ./run.sh
# analyse in merge_nscc2b_shards.sh needs every part). Same file names as
# run_all.py's collect_tables_and_figures.
#   bash scripts/ae/collect_nscc2b_partial.sh <RUN> [shard...]
# Output: <RUN>/partial/{tables,figures}.
set -uo pipefail
RUN=$(cd "$1" && pwd); shift
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source "${MORPHX_VENV:-$HOME/scratch/morphx/envs/sglang-stock}/bin/activate"
S=$RUN/shards
T=$RUN/partial/tables
F=$RUN/partial/figures
mkdir -p "$T" "$F"

copy() {  # copy <src> <dst name>
  [[ -f "$1" ]] && cp "$1" "$T/$2" && echo "table $2"
}
copy "$S/p1-llm-1-ggnn/tables/compact/llm-1-ggnn.tsv" "Figure10,12(a).txt"
copy "$S/p1-llm-4-ggnn/tables/compact/llm-4-ggnn.tsv" "Figure10,12(b).txt"
copy "$S/p1-llm-1-mm/tables/compact/llm-1-mm.tsv" "Figure11(a).txt"
copy "$S/p1-llm-4-mm/tables/compact/llm-4-mm.tsv" "Figure11(b).txt"
copy "$S/p2-Figure13/tables/Figure13.tsv" "Figure13.txt"
copy "$S/p2-Figure14/tables/Figure14.tsv" "Figure14.txt"
copy "$S/p3/tables/Figure15.tsv" "Figure15.txt"
copy "$S/p4/tables/Figure16.tsv" "Figure16.txt"
copy "$S/p4/tables/Figure16-profile-metadata.tsv" "Figure16-profile-metadata.txt"

python plot_figures.py --tables-dir "$T" --output-dir "$F" --formats png,pdf --allow-missing 2>&1 | tail -n 20
ls "$F"
