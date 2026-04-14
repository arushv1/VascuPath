#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l h_rt=4:00:00
#$ -pe omp 4
#$ -N vascupath_inference
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/inference/

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath/src

python -m inference.class5_wsi_pipeline ../data/svs/10714.svs --output ../outputs/
