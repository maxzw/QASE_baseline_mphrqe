#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --partition=gpu_shared
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.j.zwager@student.vu.nl
#SBATCH --output=job_logs/output_%A.out
#SBATCH --error=job_logs/errors_%A.err

module purge all
module load 2021
module load Anaconda3/2021.05

# define and create a unique scratch directory
SCRATCH_DIRECTORY=/global/work/${USER}/kelp/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Activate Anaconda work environment for OpenDrift
source /home/${USER}/.bashrc
source activate thesis
pip install -e .

# Run your code
srun python executables/main.py classify \
    -va "/1hop/0qual:*" \
    -va "/2hop/0qual:*" \
    -va "/3hop/0qual:*" \
    -va "/2i/0qual:*" \
    -va "/3i/0qual:*" \
    -va "/1hop-2i/0qual:*" \
    -va "/2i-1hop/0qual:*" \
    -te "/1hop/0qual:*" \
    -te "/2hop/0qual:*" \
    -te "/3hop/0qual:*" \
    -te "/2i/0qual:*" \
    -te "/3i/0qual:*" \
    -te "/1hop-2i/0qual:*" \
    -te "/2i-1hop/0qual:*" \
    --batch-size 64 \
    --use-wandb ${@:1}

    # --data-root "aifb"
    # --model-path "./saved/model_AIFB.pt"
    
    # --data-root "mutag"
    # --model-path "./saved/model_MUTAG.pt"