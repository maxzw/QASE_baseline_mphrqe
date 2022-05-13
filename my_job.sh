#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --partition=gpu_shared
#SBATCH --time=10:00:00
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
pip install .

# prepare data
srun hrqe preprocess download-wd50k
srun hrqe preprocess skip-and-download-binary

# Run your code
srun hrqe train \
    -tr "/1hop/1qual-per-triple:*" \
    -tr "/2i/1qual-per-triple:atmost40000" \
    -tr "/2hop/1qual-per-triple:40000" \
    -tr "/3hop/1qual-per-triple:40000" \
    -tr "/3i/1qual-per-triple:40000" \
    -va "/1hop/1qual-per-triple:atmost3500" \
    -va "/2i/1qual-per-triple:atmost3500" \
    -va "/2hop/1qual-per-triple:atmost3500" \
    -va "/3hop/1qual-per-triple:atmost3500" \
    -va "/3i/1qual-per-triple:atmost3500" \
    --epochs 2 \
    # --embedding-dim 192
    # --activation nn.LeakyRelu
    # --learning-rate 0.0008
    # --batch-size 64
    # --num-layers 3
    # --use-bias True
    # --graph-pooling TargetPooling \
    # --dropout 0.5 \
    # --similarity CosineSimilarity \
    # --use-wandb --wandb-name "training-example" \
    --save \
    --model-path "training-example-model.pt"

mkdir -p $HOME/QASE_baseline_mphrqe/run_data/scratch_${SLURM_JOBID} && \ 
cp -r ${SCRATCH_DIRECTORY} $HOME/QASE_baseline_mphrqe/run_data/scratch_${SLURM_JOBID}