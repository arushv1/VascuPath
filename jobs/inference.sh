#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100|L40S|A40
#$ -l h_rt=4:00:00
#$ -pe omp 4
#$ -N inference
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/inference/

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath

python -m src.inference.wsi_multi_pipeline /projectnb/rise2019/arushv/images --batch --output outputs_test/ 



