#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100|L40S|A40
#$ -l h_rt=6:00:00
#$ -pe omp 4
#$ -N train_multi
#$ -j y
#$ -m bea                        # Send email on abort, begin, and end
#$ -M arushv@bu.edu
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/training_multi

# Check job status: qstat -u arushv
# Watch live output: tail -f /projectnb/rise2019/arushv/VascuPath/logs/vascupath_train.o<JOB_ID>
# Submit job: qsub train.sh
# Check job details: qstat -j <JOB_ID>

set -e

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath/src

#python -m training.stage1_resnet --folds 5 --epochs 10
#python -m training.stage1_foundation --folds 5 --epochs 10

echo "=== GPU assigned to this job ==="
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi


python -m training.train_multi --folds 5 --epochs 30


