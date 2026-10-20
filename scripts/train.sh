#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --job-name=sft
#SBATCH --account=tml_2026
#SBATCH --partition=h200_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out

# One full fine-tuning run on one GPU, with the settings in configs/sft_config.yaml, for a dataset that make_subset.py
# wrote and listed in LLaMA-Factory/data/dataset_info.json (README step 8). A run can take several Slurm jobs of up to
# 8 hours each: when one hits its time limit, the next job on the same cluster resumes from the last checkpoint.
# Submit from scripts/ (or use submit_training.sh, which queues a whole run on several GPU types), e.g.
#   RUN=difficulty_weighted sbatch -M owl --job-name=sft_difficulty_weighted -p b200_normal_q --qos=owl_b200_normal_short train.sh
# Copies of the same run can wait in several queues. Only one trains at a time (job_lock.sh): a copy that starts while
# another is training exits right away, and one that starts after the previous job ended takes over.
# Checkpoints go to this cluster's /scratch/$USER/saves/$RUN, so only jobs on the same cluster can resume from them.
# The finished model is copied to $HOME/models/$RUN, which every ARC cluster can read.

RUN=${RUN:?set RUN=<dataset name>, e.g. RUN=difficulty_weighted}
SCRIPTS=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
FINAL=$HOME/models/$RUN

# B200 GPUs need PyTorch built for CUDA 12.8 (~/envs/sft-b200); the other GPUs use ~/envs/sft. Set ENV to use another.
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q B200; then
    ENV=${ENV:-$HOME/envs/sft-b200}
fi
ENV=${ENV:-$HOME/envs/sft}
# Use the env by putting its bin/ first on PATH, not with "source activate": if a conda env was active when you
# submitted, "source activate" inside a job falls back to the base Python without any error. Also clear the conda
# variables the job inherited.
unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
export PATH="$ENV/bin:$PATH" CONDA_PREFIX=$ENV
# DeepSpeed needs a CUDA toolkit on the node, so load the version PyTorch was built for (12.8 for the B200 env, 12.6
# otherwise). With DS_SKIP_CUDA_CHECK=1, DeepSpeed warns instead of stopping if those two CUDA versions differ.
case "$ENV" in *b200*) module load CUDA/12.8.0 ;; *) module load CUDA/12.6.0 ;; esac
export DS_SKIP_CUDA_CHECK=1
python -c "from llamafactory.train.tuner import run_exp" || { echo "training env $ENV not usable (can't import llamafactory)"; exit 1; }

# If nvidia-smi reports hardware slowdown (the GPU has cut its own clock speed, usually because it's too hot or drawing
# too much power), it runs several times slower, so quit and let a copy on another node take the run.
# nvidia-smi prints "Active" or "Not Active", so match the whole line.
if nvidia-smi --query-gpu=clocks_throttle_reasons.hw_slowdown --format=csv,noheader 2>/dev/null | grep -qE '^ *Active *$'; then
    echo "the GPU on $(hostname) is in hardware slowdown; exiting so another copy can run"
    echo "$(hostname -s)" >> "$HOME/.bad_gpu_nodes"  # later submissions skip this node (bad_nodes.sh)
    exit 1
fi

if [ -f "$FINAL/config.json" ]; then
    echo "$RUN is already finished ($FINAL); exiting."
    exit 0
fi
if [ -n "$SLURM_JOB_ID" ]; then
    source "$SCRIPTS/job_lock.sh"
    # Setting LOCK_SUFFIX (e.g. LOCK_SUFFIX=_b200) gives this copy its own lock and checkpoint folder, so it trains at
    # the same time as the other copies, for example on a faster GPU type. Whichever copy finishes first becomes the
    # final model.
    take_lock "train_$RUN${LOCK_SUFFIX:-}" || exit 0
fi
set -eo pipefail

cd "$SCRIPTS/../LLaMA-Factory"

OUT=/scratch/$USER/saves/$RUN${LOCK_SUFFIX:-}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

# EXTRA_ARGS takes more LLaMA-Factory overrides, e.g. EXTRA_ARGS="max_steps=10" for a quick test.
FORCE_TORCHRUN=1 llamafactory-cli train ../configs/sft_config.yaml \
    dataset="$RUN" \
    output_dir="$OUT" \
    tokenized_path="/scratch/$USER/tokenized/$RUN" ${EXTRA_ARGS:-}

# Copy the finished model to $HOME/models, without the checkpoints (job_lock.sh renames cut-off ones to incomplete-*).
# If a copy with a different LOCK_SUFFIX finished first, keep its model instead, since eval jobs may already use it.
if [ -f "$FINAL/config.json" ]; then
    echo "$RUN was finished first by another copy ($FINAL); keeping that model"
    exit 0
fi
mkdir -p "$FINAL"
rsync -a --exclude 'checkpoint-*' --exclude 'incomplete-*' "$OUT/" "$FINAL/"
bash "$SCRIPTS/fix_tokenizer.sh" "$FINAL"  # tokenizer files the eval env can read
echo "finished model -> $FINAL"
