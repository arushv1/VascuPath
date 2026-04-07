#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100|L40S|V100|A40
#$ -l h_rt=6:00:00
#$ -pe omp 4
#$ -N vascupath_train
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/

# Check job status: qstat -u arushv
# Watch live output: tail -f /projectnb/rise2019/arushv/VascuPath/logs/vascupath_train.o<JOB_ID>
# Submit job: qsub train.sh
# Check job details: qstat -j <JOB_ID>

set -e

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath/src

python -m training.stage1_resnet --folds 5 --epochs 10
python -m training.stage1_foundation --folds 5 --epochs 10

