"""README step 9: write scripts/eval_slices.txt, the work list for eval_worker.sh. Each line is a slice: one
fine-tuned model on one range of a benchmark's questions.

The fine-tuned models write very long answers, so a whole benchmark can take more than a day on one GPU. We cut each
benchmark into slices of about 2 to 3 A30 GPU hours (eval.py's --start_idx / --end_idx) and merge them afterwards
(merge_eval_slices.py). Slices are ordered so all models move forward together: the first slice of every benchmark
for every model, then the second, and so on. Benchmarks that already have a score are skipped.
Line format: <name> <model dir> <output dir> <benchmark> <start> <end>
Lists all five models at $HOME/models/<name>, so run it once they're all trained.
Run from eval/:  python ../scripts/make_eval_slices.py
"""
import os

HOME = os.path.expanduser("~")
RUNS = ["random_15k", "cleaned_difficulty_topic_balanced", "cleaned_difficulty_weighted", "cleaned_hardest_only", "cleaned_random"]
BENCHES = ["aime", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"]
SLICE = {"aime": 30, "amc": 40, "cn_math_2024": 30, "gpqa": 40, "kaoyan": 50, "math": 100, "minerva": 40,
         "olympiadbench": 50}  # problems per slice (AIME, AMC and CN_MATH are one slice each)


def size(bench):
    with open(f"data/{bench}/test.jsonl") as f:
        return sum(1 for line in f if line.strip())


def scored(out, bench):
    try:
        with open(f"{out}/log_{bench}.txt") as f:
            return any(line.startswith("Acc:") for line in f)
    except OSError:
        return False


def slices(bench):
    n, k = size(bench), SLICE[bench]
    return [(s, min(s + k, n)) for s in range(0, n, k)]


lines = []
for i in range(max(len(slices(b)) for b in BENCHES)):
    for bench in BENCHES:
        for run in RUNS:
            out = f"outputs/{run}"
            if not scored(out, bench) and i < len(slices(bench)):
                s, e = slices(bench)[i]
                lines.append(f"{run} {HOME}/models/{run} {out} {bench} {s} {e}")
with open("../scripts/eval_slices.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"{len(lines)} slices in ../scripts/eval_slices.txt")
