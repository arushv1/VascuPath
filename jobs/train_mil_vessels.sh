#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100|L40S
#$ -l h_rt=6:00:00
#$ -pe omp 4
#$ -N train_mil_vessels
#$ -m bea
#$ -M arushv@bu.edu
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/training_mil


# Check job status: qstat -u arushv
# Watch live output: tail -f /projectnb/rise2019/arushv/VascuPath/logs/vascupath_train.o<JOB_ID>
# Submit job: qsub train.sh
# Check job details: qstat -j <JOB_ID>

echo "=== MIL Training ==="
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

set -e

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath/

#python -m training.stage2_resnet --folds 5 --epochs 10
python -m src.ABMIL.train_eval \
    --features-dir data/processed_vessels/vessels_g \
    --comparison "all" \
    --output-dir mil_vessels/ \
    --epochs 50 \
    --lr 1e-4 \
    --patience 8 \
    --folds 5

python -m src.ABMIL.train_eval \
    --features-dir data/processed_vessels/vessels_w \
    --comparison "all" \
    --output-dir mil_vessels/ \
    --epochs 50 \
    --lr 1e-4 \
    --patience 8 \
    --folds 5

#python -m src.ABMIL.train_eval \
#    --comparison "all" \
#    --output-dir mil/ \
#    --epochs 50 \
#    --lr 1e-4 \
#    --patience 8 \
#    --folds 5
 

echo ""
echo "=== Finished at $(date) ==="