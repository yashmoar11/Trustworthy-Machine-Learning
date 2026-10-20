#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --job-name=eval_worker
#SBATCH --account=tml_2026
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out

# One eval job ("worker", README step 9). Each line of eval_slices.txt (written by make_eval_slices.py) is a slice:
# one fine-tuned model on one range of a benchmark's questions. The worker goes through the list once and runs every
# slice that isn't done yet, isn't being run by another worker, hasn't used its 3 tries, and should finish before this
# job's time limit. Each slice runs the course's eval_single.sh with only two changes: the index range (eval.py
# --start_idx / --end_idx) and where the log goes. So prompts, sampling and grading are the course eval's, and
# merge_eval_slices.py adds the slices up into a benchmark score.
# Any number of workers can run at once, on any cluster. A worker claims a slice with job_lock.sh (if a worker hits
# its time limit in the middle of a slice, a later one takes the slice over).
# Runs vLLM 0.6.1 (~/envs/evalenv) with the memory settings in vllm_fix/, on A30, L40S, A100 or H200 GPUs.
# Writes each slice's log to <output dir>/slices/, one line per finished slice to <output dir>/slices/runs.csv (read
# by merge_eval_slices.py), and how many times each slice was started to ~/.eval_slice_tries/.
# Submit from eval/ (eval_dispatcher.sh normally submits the workers), e.g.
#   sbatch -M falcon -p a30_normal_q --qos=fal_a30_normal_short --time=8:00:00 ../scripts/eval_worker.sh

cd "$SLURM_SUBMIT_DIR" || exit 1
SCRIPTS=$SLURM_SUBMIT_DIR/../scripts
SLICES=${SLICES:-$SCRIPTS/eval_slices.txt}
TRIES=$HOME/.eval_slice_tries  # how many times each slice was started
mkdir -p "$TRIES"

case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
    *B200*) echo "vLLM 0.6.1 doesn't run on B200 GPUs"; exit 1 ;;
    *H200*) KIND=h200 ;; *A100*) KIND=a100 ;; *L40S*) KIND=l40s ;; *A30*) KIND=a30 ;; *) KIND=other ;;
esac
EVALENV=$HOME/envs/evalenv
export PYTHONPATH=$SCRIPTS/vllm_fix EVAL_VLLM_FIX=1 VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE:-48}
# eval_single.sh runs "source activate $CONDA_ENV". If a conda env was active when you submitted, that falls back to
# the base Python without any error, so clear the inherited conda variables first and put the env first on PATH.
unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
export PATH="$EVALENV/bin:$PATH" CONDA_ENV=$EVALENV
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null) || { echo "eval env $EVALENV not usable (can't import vllm)"; exit 1; }

# If nvidia-smi reports hardware slowdown (the GPU has cut its own clock speed, usually because it's too hot or drawing
# too much power), it runs several times slower, so quit before claiming a slice.
# nvidia-smi prints "Active" or "Not Active", so match the whole line.
if nvidia-smi --query-gpu=clocks_throttle_reasons.hw_slowdown --format=csv,noheader 2>/dev/null | grep -qE '^ *Active *$'; then
    echo "the GPU on $(hostname) is in hardware slowdown; exiting so another worker can run"
    echo "$(hostname -s)" >> "$HOME/.bad_gpu_nodes"  # later submissions skip this node (bad_nodes.sh)
    exit 1
fi

# Rough minutes per problem on an A30 for the fine-tuned models, and each GPU type's run time relative to an A30
# (0.5 = half as long).
declare -A MIN_PER_PROBLEM=([aime]=4.5 [amc]=3.6 [cn_math_2024]=3.8 [gpqa]=4.3 [kaoyan]=2.8 [math]=1.7 [minerva]=4.0 [olympiadbench]=3.5)
declare -A SPEED=([a30]=1 [l40s]=0.9 [a100]=0.5 [h200]=0.3 [other]=1)
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
echo "$(date '+%F %T') worker on $KIND $(hostname -s), vLLM $VLLM_VERSION, slices from $SLICES, until $(date -d @$END '+%F %T')"

outfile() {  # the file eval.py writes for a slice: <output dir>/<last 3 parts of the model path>/<bench>/...
    local m=${1%/}
    echo "$2/$(basename "$(dirname "$(dirname "$m")")")/$(basename "$(dirname "$m")")/$(basename "$m")/$3/test_qwen-instruct_t0.6_k8_s$4_e$5.jsonl"
}

# The list is read on file descriptor 3, so nothing run inside the loop can read lines from it.
while read -r name model out bench s e <&3; do
    [[ -z $name || $name == \#* ]] && continue
    f=$(outfile "$model" "$out" "$bench" "$s" "$e")
    [ -f "$f" ] && continue
    slice=${name}_${bench}_s${s}_e${e}
    [ "$(cat "$TRIES/$slice" 2>/dev/null || echo 0)" -ge 3 ] && continue
    # rough run time with 30% margin plus 15 min for loading; skip slices that won't fit before the time limit
    need=$(awk "BEGIN { print int(($e - $s) * ${MIN_PER_PROBLEM[$bench]:-4.5} * 60 * ${SPEED[$KIND]} * 1.3 + 900) }")
    [ $(( $(date +%s) + need )) -gt "$END" ] && continue
    # Each slice runs in a subshell, so job_lock.sh's exit trap frees the slice's lock as soon as the slice ends.
    (
        source "$SCRIPTS/job_lock.sh"
        # if another worker has this slice, skip it without printing take_lock's messages
        take_lock "evalslice_$slice" > "/tmp/claim.$BASHPID" 2>&1 || { rm -f "/tmp/claim.$BASHPID"; exit 3; }
        cat "/tmp/claim.$BASHPID"; rm -f "/tmp/claim.$BASHPID"
        [ -f "$f" ] && exit 0  # another worker just finished it
        echo $(( $(cat "$TRIES/$slice" 2>/dev/null || echo 0) + 1 )) > "$TRIES/$slice"
        mkdir -p "$out/slices"
        export MODEL=$model OUTPUT_DIR=$out SLICE_LOG=$out/slices/log_${bench}_s${s}_e${e}.txt
        script=$(mktemp /tmp/eval_slice.XXXXXX)
        sed -e "s/^--start_idx 0 \\\\\$/--start_idx $s \\\\/" -e "s/^--end_idx -1 \\\\\$/--end_idx $e \\\\/" \
            -e 's#tee "$OUTPUT_DIR/log_${DATA}.txt"#tee "$SLICE_LOG"#' eval_single.sh > "$script"
        grep -q -- "^--start_idx $s \\\\\$" "$script" && grep -q -- "^--end_idx $e \\\\\$" "$script" \
            && grep -q 'tee "$SLICE_LOG"' "$script" || { echo "could not set the index range and log file in eval_single.sh"; exit 1; }
        bash "$SCRIPTS/fix_tokenizer.sh" "$MODEL"  # tokenizer files vLLM 0.6.1 can read
        echo "$(date '+%F %T') $slice: started (try $(cat "$TRIES/$slice"))"
        t0=$(date +%s)
        bash "$script" "$bench" > /dev/null 2>&1
        rm -f "$script"
        if [ -f "$f" ]; then
            echo "$slice,$KIND,$(hostname -s),$SLURM_CLUSTER_NAME,$SLURM_JOB_ID,vllm-$VLLM_VERSION,$t0,$(date +%s)" >> "$out/slices/runs.csv"
            echo "$(date '+%F %T') $slice: done in $(( ($(date +%s) - t0) / 60 )) min"
        else
            echo "$(date '+%F %T') $slice: FAILED after $(( ($(date +%s) - t0) / 60 )) min: $(grep -aE 'Error|error' "$SLICE_LOG" | tail -1 | cut -c1-200)"
            exit 1
        fi
    )
    # A failed slice stops this worker, so one broken GPU can't use up the 3 tries of many slices.
    [ $? -eq 1 ] && exit 1
done 3< "$SLICES"
echo "$(date '+%F %T') no slice left that fits in this job's time"
