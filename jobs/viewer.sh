#!/bin/bash -l
# Launch the WSI viewer on an interactive SCC compute node.
#
# Run this from an interactive session, NOT via qsub:
#   qrsh -P rise2019 -l h_rt=2:00:00 -pe omp 2
#   bash jobs/viewer.sh
#
# Then from your laptop terminal:
#   ssh -L 9999:<this-node>:5000 <user>@scc1.bu.edu
#   open http://localhost:5000

source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath

SLIDE_DIR="${SLIDE_DIR:-/projectnb/rise2019/JC_CTE_Images/AI export/Frontal Cortex}"
PRED_DIR="${PRED_DIR:-outputs}"
PORT="${PORT:-5000}"

echo "Compute node : $(hostname -s)"
echo "Slide dir    : $SLIDE_DIR"
echo "Pred dir     : $PRED_DIR"
echo
echo "From your laptop:"
echo "  ssh -L ${PORT}:$(hostname -s):${PORT} <user>@scc1.bu.edu"
echo "Then open http://localhost:${PORT}"
echo

python -m src.visualization.wsi_viewer \
    --slide-dir "$SLIDE_DIR" \
    --pred-dir  "$PRED_DIR" \
    --port "$PORT"
