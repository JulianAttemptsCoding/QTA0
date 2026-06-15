#!/usr/bin/env bash
# Re-launch the 5 CPCV shards (folds 20-45) for Run 2 deep, BEHIND run2.1.
# Run only AFTER run2.1 is confirmed RUNNING (so it keeps queue priority).
set -e
cd "$(dirname "$0")/.."
DATA_URI="gs://gmda-vertex-c779f701-uscentral1/repo_inputs/run2_deep_20260613"
for fr in "20 25" "25 30" "30 35" "35 40" "40 45"; do
  set -- $fr
  python vertex/train_on_vertex.py --model f4 --cv cpcv --n_groups 10 --k_test 2 \
    --epochs 12 --seeds 2 --patience 5 --lambda_rank 0.5 --demean 1 \
    --lr 0.001 --weight_decay 0.0001 --k_book 35 \
    --fold_start "$1" --fold_end "$2" \
    --run_id run2_deep_20260613 --data_uri "$DATA_URI"
done
