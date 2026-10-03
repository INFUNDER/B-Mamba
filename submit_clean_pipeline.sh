#!/bin/bash
# Submits the complete leak-free experiment grid as independent PBS jobs
# (they run in parallel whenever GPUs are free).
#   4 configs x 3 seeds = 12 runs  (~2 h each on one GPU)
# In the morning run:   python aggregate_results.py
cd "$(dirname "$0")"

if [ ! -d dataset/PraNet_Official/TrainDataset ]; then
  echo "Run 'python create_official_split.py' first."; exit 1
fi

for SEED in 42 7 2026; do
  for CFG in full baseline edge deepsup; do
    JID=$(qsub -N "bm_${CFG}_${SEED}" -v CFG=$CFG,SEED=$SEED job_clean_run.pbs)
    echo "Submitted $CFG seed=$SEED -> $JID"
  done
done
