#!/bin/bash -l
#$ -P rise2019
#$ -l h_rt=4:00:00
#$ -N normalize
#$ -m bea
#$ -M arushv@bu.edu
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/normalize/

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath/src

python -m normalization.normalize_all --input ../data/raw/train_patches/ --output ../data/norm/norm_train_patches/
