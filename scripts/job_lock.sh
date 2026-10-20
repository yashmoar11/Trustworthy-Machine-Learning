# Lets copies of one job wait in queues on several clusters while only one of them runs. train.sh takes one lock per
# training run; eval_worker.sh takes one per slice (one model on one range of a benchmark's questions), so no two eval
# jobs run the same slice. Slurm commands on a compute node only see that node's cluster, but $HOME is shared, so the
# first copy to start makes a lock folder in $HOME/.job_locks/ and, every 5 minutes, updates the time stamp of a
# "heartbeat" file in it to show it's still running. A copy that starts while the lock is held exits. A lock made less
# than a minute ago always counts as held. After that it counts as free when
#   - its owner was on this cluster and isn't pending or running anymore (e.g. the previous job hit its time limit), or
#   - its heartbeat is older than 20 minutes (the owner was on another cluster and died).
# Only one copy can take over a free lock. Source this file, then:
#   take_lock <name> || exit 0      # the lock goes away when this shell exits (the job, or a subshell)
# A job killed by its time limit can't remove its lock, so the next copy that starts takes it over.

# Is job $1 on this cluster still pending (PD) or running (R)? squeue keeps showing a job for about 5 minutes after
# it ends, in states like CG (completing), TO (timed out) or CD (completed), so only ask for PD and R. If squeue fails,
# count the job as alive, unless the error says "Invalid job id" (Slurm no longer knows the job).
job_alive() {
    local out
    out=$(squeue -h -j "$1" -t PD,R -o %i 2>&1) && { [ -n "$out" ]; return; }
    [[ $out != *"Invalid job id"* ]]
}

take_lock() {
    LOCK=$HOME/.job_locks/$1
    mkdir -p "$HOME/.job_locks"
    if ! mkdir "$LOCK" 2>/dev/null; then
        local owner_job owner_cluster
        { read -r _ owner_job _ owner_cluster _ < "$LOCK/owner"; } 2>/dev/null  # "job <id> on <cluster> (...)"
        if [ "$owner_cluster" = "$SLURM_CLUSTER_NAME" ] && [ "$owner_job" = "$SLURM_JOB_ID" ]; then
            echo "this job was requeued (it owned the lock of $1 before); continuing"
        elif [ -n "$(find "$LOCK" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then
            # The lock folder was made less than a minute ago, so the copy that made it may still be writing the
            # owner file.
            echo "another copy of $1 just took the lock; exiting."
            return 1
        elif [ "$owner_cluster" = "$SLURM_CLUSTER_NAME" ] && [ -n "$owner_job" ]; then
            # same cluster: ask Slurm instead of trusting the heartbeat
            if job_alive "$owner_job"; then
                echo "another copy of $1 is running ($(cat "$LOCK/owner" 2>/dev/null)); exiting."
                return 1
            fi
            echo "the previous copy of $1 (job $owner_job on this cluster) has ended; taking over"
        elif [ -n "$(find "$LOCK/heartbeat" -mmin -20 2>/dev/null)" ]; then
            echo "another copy of $1 is running ($(cat "$LOCK/owner" 2>/dev/null)); exiting."
            return 1
        else
            echo "taking over the lock of $1, its heartbeat is over 20 minutes old or missing ($(cat "$LOCK/owner" 2>/dev/null))"
        fi
        # Only one copy may take over: the one whose mkdir of $LOCK.takeover works (mkdir fails if the folder exists).
        # A $LOCK.takeover folder older than 5 minutes was left by a copy that crashed, so remove it first.
        [ -d "$LOCK.takeover" ] && [ -z "$(find "$LOCK.takeover" -maxdepth 0 -mmin -5)" ] && rmdir "$LOCK.takeover"
        mkdir "$LOCK.takeover" 2>/dev/null || { echo "another copy is taking over $1; exiting."; return 1; }
    fi
    echo "job $SLURM_JOB_ID on ${SLURM_CLUSTER_NAME:-?} ($(hostname)), started $(date)" > "$LOCK/owner"
    touch "$LOCK/heartbeat"
    rmdir "$LOCK.takeover" 2>/dev/null
    (while sleep 300; do touch "$LOCK/heartbeat"; done) </dev/null >/dev/null 2>&1 &
    HEARTBEAT_PID=$!
    trap 'pkill -P $HEARTBEAT_PID 2>/dev/null; kill $HEARTBEAT_PID 2>/dev/null; rm -rf "$LOCK"' EXIT
    # For training locks only. If a job was killed by its time limit while saving a checkpoint, that checkpoint has no
    # trainer_state.json (it's saved last) and resuming from it fails, so rename it and resume from the one before.
    # (train.sh saves to /scratch/$USER/saves/<lock name without "train_">.)
    if [[ $1 == train_* ]]; then
        local saves=/scratch/$USER/saves/${1#train_} step
        for step in $(ls -d "$saves"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -rn); do
            [ -f "$saves/checkpoint-$step/trainer_state.json" ] && break
            echo "checkpoint-$step is incomplete (the previous job was stopped while saving it); renaming it to incomplete-$step"
            mv "$saves/checkpoint-$step" "$saves/incomplete-$step"
        done
    fi
    # Copies still waiting in other queues stay there on purpose: if this one fails, they take over, and once the
    # work is done they exit right after starting.
    return 0
}
