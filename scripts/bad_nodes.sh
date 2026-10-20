#!/bin/bash
# Print an --exclude option for the nodes whose GPU was caught in hardware slowdown, i.e. running at a cut clock speed
# (train.sh and eval_worker.sh write them to ~/.bad_gpu_nodes), so new jobs skip them. It only prints the given
# cluster's nodes:  sbatch $(bash bad_nodes.sh tinkercliffs) ...
case "$1" in tinkercliffs) pre=tc- ;; falcon) pre=fal ;; owl) pre=owl ;; *) exit 0 ;; esac
nodes=$(grep -s "^$pre" "$HOME/.bad_gpu_nodes" | sort -u | paste -sd, -)
[ -n "$nodes" ] && echo "--exclude=$nodes"
