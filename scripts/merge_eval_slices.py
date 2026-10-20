"""README step 9: add finished eval slices up into benchmark scores, graded with the course's grading code.

A slice is one fine-tuned model's answers to one range of a benchmark's questions, made by eval_worker.sh. For every
model and benchmark in the slice list that has no score yet and whose slices are all done, this grades every saved
answer again with the course's code (the same calls eval.py makes) and writes <output dir>/log_<benchmark>.txt with
eval.py's score lines, so collect_results.py reads it like a normal run. eval.py prints two scores: Acc, the share of
questions where at least one of the 8 answers is right (we call it pass@8), and Pass@1, the share of right answers.
Both are averages over questions, so adding the slices up gives the same numbers as one full run. If a log without a
score is already at that path (left by a full eval_single.sh run that didn't finish), it's moved to
<output dir>/crashed/ first. Prints how far along every model and benchmark is.
Run from eval/ with the eval env:  ~/envs/evalenv/bin/python ../scripts/merge_eval_slices.py [slice lists...]
(the default list is ../scripts/eval_slices.txt)
"""
import collections
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import time
import warnings
from math import comb

warnings.filterwarnings("ignore")  # the grader's sympy calls print deprecation warnings
sys.path.insert(0, os.getcwd())
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt  # noqa: E402,F401  (eval.py's imports)
from utils.parser import *  # noqa: E402,F401,F403
from utils.data_loader import load_data  # noqa: E402
from utils.math_normalization import *  # noqa: E402,F401,F403
from utils.grader import *  # noqa: E402,F401,F403

K = 1  # eval_single.sh runs eval.py with --k 1


def outfile(model, out, bench, s, e):
    model_name = "/".join(model.rstrip("/").split("/")[-3:])
    return f"{out}/{model_name}/{bench}/test_qwen-instruct_t0.6_k8_s{s}_e{e}.jsonl"


def grade_problems(path, bench, s, e):
    """eval.py's grading loop over one slice. Per problem: (any answer correct, pass@1, answers graded differently
    than in the slice's own file)."""
    examples = load_data(bench, "test", "./data")[s:e]
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == len(examples), f"{path}: {len(rows)} rows for {len(examples)} problems"
    result = []
    for d, row in zip(examples, rows):
        assert row["question"] == parse_question(d, bench), f"{path}: problems out of order"
        gt_cot, gt_ans = parse_ground_truth(d, bench)
        with contextlib.redirect_stdout(io.StringIO()):  # the grader prints an ANTLR version warning on every parse
            generated_answers = [extract_answer(r, bench) for r in row["generated_responses"]]
            is_correct_list = [check_is_correct(a, gt_ans) for a in generated_answers]
        changed = sum(a != b for a, b in zip(is_correct_list, row["answers_correctness"]))
        pass_at_k = 0
        if len(is_correct_list) > 1:
            c, n = sum(is_correct_list), len(generated_answers)
            if c > 0:
                pass_at_k = 1 if n - c < K else 1 - comb(n - c, K) / comb(n, K)
        result.append((any(is_correct_list), pass_at_k, changed))
    return result


def grade(path, bench, s, e):
    """(problems with a correct answer, sum of pass@1, answers graded differently, problems) for one slice"""
    problems = grade_problems(path, bench, s, e)
    return (sum(p[0] for p in problems), sum(p[1] for p in problems), sum(p[2] for p in problems), len(problems))


def runs_info(out):
    """slice id -> how it was generated, from the workers' runs.csv"""
    info = {}
    try:
        for slice_id, kind, host, cluster, job, vllm, t0, t1 in csv.reader(open(f"{out}/slices/runs.csv")):
            info[slice_id] = f"{vllm}/{kind}/{host}"
    except OSError:
        pass
    return info


def merge(slices_file):
    groups = collections.OrderedDict()
    for line in open(slices_file):
        if line.strip() and not line.startswith("#"):
            name, model, out, bench, s, e = line.split()
            groups.setdefault((name, model, out, bench), []).append((int(s), int(e)))
    rows = []
    for (name, model, out, bench), slices in groups.items():
        log = f"{out}/log_{bench}.txt"
        done = [os.path.exists(outfile(model, out, bench, s, e)) for s, e in slices]
        score = next((line.strip() for line in open(log) if line.startswith("Acc:")), None) if os.path.exists(log) else None
        if score is None and all(done):
            info = runs_info(out)
            total = [0, 0.0, 0, 0]
            for s, e in slices:
                for i, v in enumerate(grade(outfile(model, out, bench, s, e), bench, s, e)):
                    total[i] += v
            correct_cnt, pass_sum, changed, n = total
            if os.path.exists(log):
                os.makedirs(f"{out}/crashed", exist_ok=True)
                shutil.move(log, f"{out}/crashed/log_{bench}_{time.strftime('%m%d_%H%M', time.localtime(os.path.getmtime(log)))}.txt")
            with open(log, "w") as f:
                f.write(f"merged from {len(slices)} slices, graded again with the course grader in this env "
                        f"({changed} answers graded differently than in the slice files):\n")
                f.write(" ".join(f"s{s}_e{e}[{info.get(f'{name}_{bench}_s{s}_e{e}', '?')}]" for s, e in slices) + "\n")
                f.write(f"correct cnt / total cnt: {correct_cnt}/{n}\n")
                f.write(f"Acc: {correct_cnt / n:.4f}\n")
                f.write(f"Pass@{K}: {pass_sum}/{n} = {pass_sum / n:.4f}\n")
            score = f"Acc: {correct_cnt / n:.4f} Pass@1: {pass_sum / n:.4f} (merged, {changed} regraded differently)"
        rows.append(f"  {name:<20} {bench:<14} {score or f'{sum(done)}/{len(slices)} slices done'}")
    print(f"{slices_file}:")
    print("\n".join(rows))


if __name__ == "__main__":
    for path in sys.argv[1:] or ["../scripts/eval_slices.txt"]:
        if os.path.exists(path):
            merge(path)
