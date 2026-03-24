#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l h_rt=4:00:00
#$ -pe omp 4
#$ -N vascupath_train
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath/src

python -m training.class5_foundation --folds 3 --epochs 10