#!/bin/bash
# Queue one training run (README step 8) on every GPU type that fits the model: A100 and H200 on TinkerCliffs, B200 on
# Owl. Run it on a LOGIN node, since only login nodes can submit to other clusters. For each partition below it queues
# a chain of 8 hour train.sh jobs, each starting after the one before it ends. Only one copy trains at a time
# (job_lock.sh), and later jobs on the same cluster resume from its checkpoints.
#   bash submit_training.sh cleaned_difficulty_weighted
RUN=${1:?usage: submit_training.sh <dataset name>}
cd "$(dirname "$0")" || exit 1
mkdir -p logs

chain() {  # chain <cluster> <number of jobs> <sbatch options...>
    local cluster=$1 n=$2; shift 2
    local dep="" id
    for ((i = 1; i <= n; i++)); do
        id=$(RUN=$RUN sbatch --parsable -M "$cluster" --job-name="sft_$RUN" $(bash bad_nodes.sh "$cluster") $dep "$@" train.sh | cut -d';' -f1)
        echo "  $cluster $* job $i of $n -> job id $id"
        dep="--dependency=afterany:$id"
    done
}
# 4 jobs in the normal A100 partition (the slowest GPU here), 2 in the others. Jobs in the preemptable partitions can
# be stopped early when other jobs need the GPUs.
chain tinkercliffs 4 -p a100_normal_q --qos=tc_a100_normal_short
chain tinkercliffs 2 -p h200_normal_q --qos=tc_h200_normal_short
chain tinkercliffs 2 -p a100_preemptable_q --qos=tc_a100_preemptable_base
chain tinkercliffs 2 -p h200_preemptable_q --qos=tc_h200_preemptable_base
chain owl 2 -p b200_normal_q --qos=owl_b200_normal_short
