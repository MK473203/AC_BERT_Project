#!/bin/bash
#SBATCH --job-name=AnyDepthTrain
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

echo "Load Anaconda module"
module load anaconda/2024.06

#echo "Printing all environment variables"
#printenv

LOCAL_SCRATCH="/tmp/$USER/Project"
mkdir -p $LOCAL_SCRATCH

echo "Verifying LOCAL_SCRATCH directory"
if [ ! -d "$LOCAL_SCRATCH" ]; then
  echo "Error: LOCAL_SCRATCH directory does not exist."
  exit 1
else
  echo "LOCAL_SCRATCH directory exists."
fi

echo "Copy data from home directory"
cp -r $HOME/Documents/BERT $LOCAL_SCRATCH/
cd $LOCAL_SCRATCH/

echo "Creating directory for packages"
mkdir -p $LOCAL_SCRATCH/packages
mkdir -p $LOCAL_SCRATCH/envs

export TMPDIR=$LOCAL_SCRATCH/tmp
export TEMP=$LOCAL_SCRATCH/tmp
export TMP=$LOCAL_SCRATCH/tmp
mkdir -p $TMPDIR

# Clean previous environment if exists
echo "Creating new environment"
conda clean --all

echo "Configure Conda to use local scratch space"
conda config --add pkgs_dirs $LOCAL_SCRATCH/packages
conda config --add envs_dirs $LOCAL_SCRATCH/envs

echo "Create and activate environment"
cd BERT/
conda env create --file bertenvironment.yml --prefix $LOCAL_SCRATCH/envs/Bert

echo "Dataset and Environment Prepared"
