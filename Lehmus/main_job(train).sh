#!/bin/bash
#SBATCH --job-name=AnyDepthTrain
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH --time=12:00:00


echo "Load Anaconda module"
module load anaconda/2024.06

LOCAL_SCRATCH="/tmp/$USER/Project"
mkdir -p $LOCAL_SCRATCH
cd $LOCAL_SCRATCH/
cd BERT/
conda activate /tmp/akorpi20/Project/envs/Bert
echo "Starting Training"
ipython BERT-PROJECT_DATASET.py

echo "Processing finished"
