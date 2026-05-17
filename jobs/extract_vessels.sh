#!/bin/bash -l
#$ -P rise2019
#$ -l gpus=1
#$ -l gpu_type=A100|L40S
#$ -l h_rt=10:00:00
#$ -pe omp 4
#$ -N extract_vessels
#$ -m bea
#$ -M arushv@bu.edu
#$ -j y
#$ -o /projectnb/rise2019/arushv/VascuPath/logs/extract/

module load cuda
source /projectnb/rise2019/arushv/VascuPath/vascuenv/bin/activate
cd /projectnb/rise2019/arushv/VascuPath

python -m src.ABMIL.extract_features_vessel --svs-dir "/projectnb/rise2019/JC_CTE_Images/AI export/Frontal Cortex" --output-dir data/processed_vessels/ --batch-size 64 --num-workers 4 --resume


