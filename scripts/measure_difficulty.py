r"""README step 6: see how often the base model (Qwen2.5-3B-Instruct, before any fine-tuning) already solves each
problem.

The course README (reds-lab/Project-Reasoning-SFT-LLM) suggests measuring a problem's difficulty by generating several
answers and checking how many are right. We do that with the model we fine-tune: 4 tries per problem with the course
eval's system prompt and sampling (temperature 0.6, top-p 0.95), answers capped at 4,096 tokens, and problems whose
prompt is longer than 4,096 tokens skipped. Each try's final answer is pulled out with the eval's extract_answer and
checked against the final answer of the row's own solution: option letters must match exactly, two plain numbers must
agree within a relative tolerance of 1e-4, and anything else goes to the eval's check_is_correct. We only use problems
whose solution's final answer is a plain number or an option letter, since that comparison is reliable. The problem
list comes from `select_subsets.py --candidates`, and select_subsets.py uses the results as solve rates.

The problems are split into shards, one GPU job each: with --num_shards 6, shard i gets every 6th problem, starting
at problem i. Use the eval env (README step 1) and run from scripts/. We ran 6 shards in parallel on Falcon A30s,
about an hour each:
    for i in 0 1 2 3 4 5; do
        sbatch -M falcon -A tml_2026 -p a30_normal_q --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 \
            --output=logs/difficulty_%j.out --wrap="$HOME/envs/evalenv/bin/python measure_difficulty.py --shard $i --num_shards 6"
    done
Each shard writes eval/train_data/difficulty_<shard>.parquet (row, n_correct, n_samples, mean_tokens) and
difficulty_answers_<shard>.parquet (every extracted answer). --regrade grades the answers in that second file again
(CPU only) and rewrites difficulty_<shard>.parquet, with mean_tokens left empty.
"""
import argparse
import importlib.util
import math
import os
import re
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval"
DATA = EVAL / "train_data"
MODEL = EVAL / "models" / "Qwen2.5-3B-Instruct"
SAMPLES, MAX_TOKENS = 4, 4096

sys.path.insert(0, str(EVAL))
os.chdir(EVAL)
from utils.grader import check_is_correct  # noqa: E402
from utils.parser import extract_answer, strip_string  # noqa: E402


def eval_system_prompt():
    spec = importlib.util.spec_from_file_location("p", EVAL / "prompts" / "qwen-instruct" / "math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.system_prompt


def option_letter(ans):
    return re.sub(r"\\text(?:bf)?\{([^}]*)\}", r"\1", ans).strip().strip("()").strip()


def _alarm(signum, frame):
    raise TimeoutError


def as_number(ans):
    try:
        return float(strip_string(ans).replace(",", ""))
    except (ValueError, TypeError):
        return None


def grade(args):
    pred, ref = args
    if not pred:
        return False
    if re.fullmatch(r"[A-E]", ref):
        return option_letter(pred) == ref
    if strip_string(pred) == strip_string(ref):
        return True
    p, r = as_number(pred), as_number(ref)
    if p is not None and r is not None:  # two plain numbers: relative tolerance 1e-4 (the eval's grader uses an absolute one)
        return math.isclose(p, r, rel_tol=1e-4)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(5)
    try:
        return bool(check_is_correct(pred, ref, timeout=False))
    except Exception:
        return False
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--regrade", action="store_true", help="grade this shard's saved answers again (CPU only)")
    args = parser.parse_args()

    if args.regrade:
        ans = pd.read_parquet(DATA / f"difficulty_answers_{args.shard}.parquet")
        with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
            ok = list(pool.map(grade, zip(ans.answer, ans.teacher_answer), chunksize=64))
        ans["ok"] = ok
        res = ans.groupby("row", sort=False).ok.agg(n_correct="sum", n_samples="size").reset_index()
        res["mean_tokens"] = float("nan")  # lengths aren't saved with the answers
        res.to_parquet(DATA / f"difficulty_{args.shard}.parquet")
        print(f"regraded shard {args.shard}: {res.n_correct.value_counts().sort_index().to_dict()}", flush=True)
        raise SystemExit

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cand = pd.read_parquet(DATA / "difficulty_candidates.parquet").iloc[args.shard :: args.num_shards]
    tok = AutoTokenizer.from_pretrained(MODEL)
    system = eval_system_prompt()
    prompts = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q in cand.instruction]
    fits = [len(tok(p).input_ids) <= 8192 - MAX_TOKENS for p in prompts]
    cand, prompts = cand[fits], [p for p, f in zip(prompts, fits) if f]
    print(f"shard {args.shard}/{args.num_shards}: {len(prompts):,} problems x {SAMPLES} attempts", flush=True)

    llm = LLM(model=str(MODEL), gpu_memory_utilization=0.92, max_model_len=8192, seed=0)
    outs = llm.generate(prompts, SamplingParams(n=SAMPLES, temperature=0.6, top_p=0.95, max_tokens=MAX_TOKENS, seed=0))

    tasks, lengths = [], []
    for o, ref in zip(outs, cand.teacher_answer):
        tasks += [(extract_answer(c.text), ref) for c in o.outputs]
        lengths.append(sum(len(c.token_ids) for c in o.outputs) / len(o.outputs))
    # Save the extracted answers so grading can be redone without generating again.
    pd.DataFrame({"row": [r for r in cand.row.values for _ in range(SAMPLES)], "answer": [p for p, _ in tasks],
                  "teacher_answer": [g for _, g in tasks]}).to_parquet(DATA / f"difficulty_answers_{args.shard}.parquet")
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        ok = list(pool.map(grade, tasks, chunksize=64))

    res = pd.DataFrame({"row": cand.row.values,
                        "n_correct": [sum(ok[i * SAMPLES : (i + 1) * SAMPLES]) for i in range(len(cand))],
                        "n_samples": SAMPLES, "mean_tokens": lengths})
    out = DATA / f"difficulty_{args.shard}.parquet"
    res.to_parquet(out)
    print(f"problems by number of right tries: {res.n_correct.value_counts().sort_index().to_dict()} -> {out}", flush=True)
