#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --job-name=eval_dispatcher
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out

# A small CPU job (README step 9) that keeps eval workers (eval_worker.sh jobs) queued on A30, L40S, A100 and H200 GPUs
# while some slices in eval_slices.txt (one model on one range of a benchmark's questions, see make_eval_slices.py)
# haven't been taken by a worker yet. Every 15 minutes it also adds finished slices up into scores
# (merge_eval_slices.py). A compute node can only submit to its own cluster, so Falcon workers go through a
# TinkerCliffs login node (ssh between ARC nodes needs no password). Stops once every slice is done or has used up its
# 3 tries.
# Submit from eval/:  sbatch -M tinkercliffs ../scripts/eval_dispatcher.sh
# One pass that prints the sbatch commands instead of running them (merge_eval_slices.py still runs and may write
# score logs):  DRY=1 ONCE=1 bash ../scripts/eval_dispatcher.sh

# One line per kind of worker job (a "lane"): a name (its jobs are called evw_<name>), the cluster, partition and QoS
# (Slurm's "quality of service", which sets job limits and priority), the time limit in hours, and the most jobs of
# this kind to keep queued or running (never more than the number of unclaimed slices). Each GPU type gets a lane
# for both of its QoS: the _short QoS caps how many jobs one user can run (10 on A30, L40S and A100, 5 on H200), and
# jobs in the _base QoS don't count toward that cap.
LANES=(  # name cluster partition qos hours workers
    "a30    falcon       a30_normal_q  fal_a30_normal_short  8 10"
    "a30b   falcon       a30_normal_q  fal_a30_normal_base   8 10"
    "l40s   falcon       l40s_normal_q fal_l40s_normal_short 8 10"
    "l40sb  falcon       l40s_normal_q fal_l40s_normal_base  8 6"
    "a100   tinkercliffs a100_normal_q tc_a100_normal_short  8 10"
    "a100b  tinkercliffs a100_normal_q tc_a100_normal_base   8 6"
    "h200   tinkercliffs h200_normal_q tc_h200_normal_short  8 5"
    "h200b  tinkercliffs h200_normal_q tc_h200_normal_base   8 4"
)
LOGINS="tinkercliffs1 tinkercliffs2"
SLICES=../scripts/eval_slices.txt

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
mkdir -p logs
for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF); do unset "$v"; done  # so the jobs this one submits don't inherit its Slurm settings
PY=$HOME/envs/evalenv/bin/python

on_login() {  # run a command on the first login node that answers
    local host rc
    for host in $LOGINS; do
        ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" "cd $PWD && $1"
        rc=$?
        [ $rc -ne 255 ] && return $rc  # ssh exits with 255 when ssh itself fails (e.g. it can't reach this node)
    done
    return 255
}

count_slices() {  # prints: <done> <held by a worker> <waiting> <given up>
    "$PY" - "$SLICES" <<'EOF'
import os, sys, time
home = os.path.expanduser("~")

def held(lock):  # a worker still holds this lock: its heartbeat (or the lock itself) changed in the last 20 minutes
    for p in (f"{lock}/heartbeat", lock):
        try:
            if time.time() - os.path.getmtime(p) < 20 * 60:
                return True
        except OSError:
            pass
    return False

n = [0, 0, 0, 0]
for line in open(sys.argv[1]):
    if not line.strip() or line.startswith("#"):
        continue
    name, model, out, bench, s, e = line.split()
    slice_id = f"{name}_{bench}_s{s}_e{e}"
    f = f"{out}/{'/'.join(model.rstrip('/').split('/')[-3:])}/{bench}/test_qwen-instruct_t0.6_k8_s{s}_e{e}.jsonl"
    try:
        tries = int(open(f"{home}/.eval_slice_tries/{slice_id}").read())
    except (OSError, ValueError):
        tries = 0
    lock = f"{home}/.job_locks/evalslice_{slice_id}"
    if os.path.exists(f):
        n[0] += 1
    elif os.path.isdir(lock) and held(lock):  # a lock left by a worker that died doesn't count, so the slice waits again
        n[1] += 1
    elif tries >= 3:
        n[3] += 1
    else:
        n[2] += 1
print(*n)
EOF
}

while true; do
    "$PY" ../scripts/merge_eval_slices.py
    read -r n_done n_claimed n_free n_failed < <(count_slices)
    echo "$(date '+%F %T') slices: $n_done done, $n_claimed held by a worker, $n_free waiting, $n_failed given up after 3 tries"
    [ $((n_claimed + n_free)) -eq 0 ] && { echo "no slices left to run"; break; }
    for lane in "${LANES[@]}"; do
        read -r name cluster partition qos hours workers <<< "$lane"
        want=$(( workers < n_free ? workers : n_free ))
        if [ "$cluster" = tinkercliffs ]; then
            queued=$(squeue -h -u "$USER" -n "evw_$name") || continue
        else
            queued=$(on_login "squeue -M $cluster -h -u $USER -n evw_$name") || continue
        fi
        have=$(grep -c . <<< "$queued")
        for _ in $(seq $((want - have))); do
            cmd="sbatch -M $cluster -p $partition --qos=$qos $(bash ../scripts/bad_nodes.sh "$cluster") --time=$hours:00:00 --job-name=evw_$name ../scripts/eval_worker.sh"
            [ -n "$DRY" ] && { echo "would run: $cmd"; continue; }
            if [ "$cluster" = tinkercliffs ]; then out=$(eval "$cmd" 2>&1); else out=$(on_login "$cmd" 2>&1); fi
            echo "$(date '+%F %T') [$name] $out"
        done
    done
    [ -n "$ONCE" ] && break
    sleep 900
done
