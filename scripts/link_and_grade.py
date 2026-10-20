r"""README step 5, part 2: find each math row in its source dataset and grade the solution's final answer.

The math problems in our training data (the course's acereason11_100k, taken from NVIDIA's AceReason-1.1-SFT) come
from two public datasets, nvidia/OpenMathReasoning (OMR) and AI-MO/NuminaMath-CoT. Matching the prompt text gives us a
reference answer, where the problem came from (forum, competition, ...), and for OMR a difficulty score (the share of
32 tries that Qwen2.5-72B with tools got right). Both are read straight from the Hugging Face Hub: from OMR only its
small metadata columns, from NuminaMath-CoT also its solutions, since its reference answer is the \boxed{} answer
inside each one. The final answer of each row's own solution is pulled out with the course eval's extract_answer and
graded with its check_is_correct (two option letters are compared directly, see grade()).

CPU only, needs internet. Run from scripts/ with the eval env (README step 1; it has the grader's packages):
    sbatch -A tml_2026 -p normal_q --cpus-per-task=32 --mem=64G --time=2:00:00 --output=logs/link_%j.out \
        --wrap="$HOME/envs/evalenv/bin/python link_and_grade.py"
Writes eval/train_data/linked.parquet (one row per math row). For a quick test, set the environment variable
LIMIT=500: it then grades only the first 500 math rows against 2 of the OMR files and writes linked.test.parquet.
"""
import os
import re
import signal
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval" / "train_data" / "acereason11_100k" / "data"
OUT = REPO / "eval" / "train_data" / "linked.parquet"
NUM_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

sys.path.insert(0, str(REPO / "eval"))
os.chdir(REPO / "eval")
from utils.grader import check_is_correct  # noqa: E402
from utils.parser import extract_answer, strip_string  # noqa: E402

OMR_COLS = ["problem", "expected_answer", "problem_source", "problem_type", "pass_rate_72b_tir"]
# A multiple choice options block like "A. 5 B. 7 C. 9" or "(A) 5 (B) 7 (C) 9", with at least options A to C.
MCQ = re.compile(r"(?:^|\s|\()A[.):：]\s?.+?(?:^|\s|\()B[.):：]\s?.+?(?:^|\s|\()C[.):：]", re.S)
CJK = re.compile(r"[一-鿿]")
LETTER = re.compile(r"^[A-E]$")
# Answers a rule-based grader can't judge well: sentences ("The graph is not planar") and logic statements.
WORDY = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{5,}|\\(?:Rightarrow|iff|implies|forall|exists|blacksquare)|Q\.E\.D")


def norm(text):
    return " ".join(text.split())


def read_omr_file(args):
    fs, path, wanted = args
    table = pq.read_table(fs.open(path, block_size=8 * 2**20), columns=OMR_COLS).to_pandas()
    table = table.drop_duplicates("problem")
    table["key"] = table.problem.map(norm)
    hits = table[table.key.isin(wanted)].drop(columns="problem")
    return len(table), hits


def clean_ref(ans):
    """Drop an appended approximation, "2\\sqrt{14} \\approx 7.48" -> "2\\sqrt{14}", and math delimiters."""
    if not ans:
        return ans
    ans = ans.split("\\approx")[0].strip()
    # OMR wraps a lot of answers in "\( ... \)" or "\[ ... \]"
    ans = re.sub(r"^\\[(\[]\s*|\s*\\[)\]]$", "", ans).strip()
    return ans.strip("$").strip()


def option_letter(ans):
    """"\\text{(B)}", "(B)", "B" -> "B". Anything else comes back stripped."""
    return re.sub(r"\\text(?:bf)?\{([^}]*)\}", r"\1", ans).strip().strip("()").strip()


def gradability(pred, gt):
    """'letters' if both are option letters, 'math' if neither is and the reference is short (60 characters or less)
    with no words or logic symbols, else None."""
    if not pred or not gt:
        return None
    p_letter, g_letter = LETTER.match(option_letter(pred)), LETTER.match(option_letter(gt))
    if p_letter and g_letter:
        return "letters"
    if p_letter or g_letter or len(gt) > 60 or WORDY.search(gt):
        return None
    return "math"


def _alarm(signum, frame):
    raise TimeoutError


def grade(pred, gt, how):
    """The eval's check_is_correct, except that two option letters are compared directly. True / False, or None if
    the grader fails or takes longer than 5 s."""
    if how == "letters":
        return option_letter(pred) == option_letter(gt)
    if strip_string(pred) == strip_string(gt):
        return True
    # timeout=False skips the grader's own timeout, which starts a new process for every sympy check. The alarm caps
    # each check at 5 s instead.
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(5)
    try:
        return bool(check_is_correct(pred, gt, timeout=False))
    except Exception:
        return None
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    # Same file order as datasets.load_dataset in the other scripts, so "row" numbers match everywhere.
    shards = sorted(SRC.glob("train-*.parquet"))
    rows = pd.concat(
        (pq.read_table(p, columns=["instruction", "output", "extra_info"]).to_pandas() for p in shards),
        ignore_index=True,
    )
    rows["row"] = range(len(rows))
    rows["source"] = rows.extra_info.map(lambda e: e["source"])
    rows = rows[rows.extra_info.map(lambda e: e["category"]) == "math"].drop(columns="extra_info")
    limit = int(os.environ.get("LIMIT", 0))  # test run: LIMIT=500 grades 500 rows against 2 OMR files
    if limit:
        rows = rows.head(limit)
    rows["key"] = rows.instruction.map(norm)
    wanted = set(rows.key)
    print(f"math rows: {len(rows):,}, unique prompts: {len(wanted):,}", flush=True)

    fs = HfFileSystem()
    omr_files = sorted(fs.glob("datasets/nvidia/OpenMathReasoning/data/cot-*.parquet"))[: 2 if limit else None]
    print(f"reading metadata columns from {len(omr_files)} OpenMathReasoning files...", flush=True)
    omr_hits, n_problems = [], 0
    with ThreadPoolExecutor(8) as pool:
        for i, (n, hits) in enumerate(pool.map(read_omr_file, [(fs, p, wanted) for p in omr_files])):
            n_problems += n
            omr_hits.append(hits)
            if i % 24 == 0:
                print(f"  {i + 1}/{len(omr_files)} files", flush=True)
    omr = pd.concat(omr_hits).drop_duplicates("key").set_index("key")
    print(f"OMR: {n_problems:,} problem rows scanned, {len(omr):,} of our prompts found", flush=True)

    numina_files = sorted(fs.glob("datasets/AI-MO/NuminaMath-CoT/data/train-*.parquet"))
    numina = pd.concat(
        pq.read_table(fs.open(p, block_size=8 * 2**20), columns=["source", "problem", "solution"]).to_pandas()
        for p in numina_files
    )
    numina["key"] = numina.problem.map(norm)
    numina = numina[numina.key.isin(wanted)].drop_duplicates("key").set_index("key")
    numina["ref_answer"] = numina["solution"].map(lambda s: extract_answer(s) if "boxed" in s else "")
    print(f"NuminaMath-CoT: {len(numina):,} of our prompts found", flush=True)

    out = pd.DataFrame({"row": rows.row.values, "source": rows.source.values})
    keys = rows.key.values
    out["omr_problem_source"] = [omr["problem_source"].get(k) for k in keys]
    out["omr_problem_type"] = [omr["problem_type"].get(k) for k in keys]
    out["pass_rate_72b"] = pd.to_numeric([omr["pass_rate_72b_tir"].get(k) for k in keys], errors="coerce")
    out["numina_source"] = [numina["source"].get(k) for k in keys]
    gt_omr = [omr["expected_answer"].get(k) for k in keys]
    gt_num = [numina["ref_answer"].get(k) for k in keys]
    out["gt_answer"] = [clean_ref(a) if a else (clean_ref(b) if b else None) for a, b in zip(gt_omr, gt_num)]
    out["gt_from"] = ["omr" if a else ("numina" if b else None) for a, b in zip(gt_omr, gt_num)]

    outputs = rows.output.values
    out["pred_answer"] = [extract_answer(o) for o in outputs]  # the final answer of the row's own solution
    think = [o.partition("</think>")[0] for o in outputs]
    answer = [o.partition("</think>")[2] for o in outputs]
    out["n_boxed_answer"] = [a.count("\\boxed") for a in answer]
    out["mcq"] = [bool(MCQ.search(p)) for p in rows.instruction.values]
    out["cjk_output"] = [len(CJK.findall(o)) for o in outputs]
    # zlib compressed size / raw size of the reasoning. Low means repetitive, looping text.
    out["think_zlib_ratio"] = [len(zlib.compress(t.encode(), 6)) / max(len(t.encode()), 1) for t in think]

    print("grading the solutions' final answers against the reference answers...", flush=True)
    out["gradable"] = [gradability(p, g) for p, g in zip(out.pred_answer, out.gt_answer)]
    todo = [(i, p, g, how) for i, (p, g, how) in enumerate(zip(out.pred_answer, out.gt_answer, out.gradable)) if how]
    correct = [None] * len(out)
    with ProcessPoolExecutor(NUM_PROC) as pool:
        futures = [(i, pool.submit(grade, p, g, how)) for i, p, g, how in todo]
        for n, (i, fut) in enumerate(futures):
            try:
                correct[i] = fut.result(timeout=60)
            except Exception:
                correct[i] = None
            if n % 10000 == 0:
                print(f"  graded {n:,}/{len(todo):,}", flush=True)
        out["correct"] = correct
        saved = OUT.with_suffix(".test.parquet") if limit else OUT
        out.to_parquet(saved)
        print(f"saved -> {saved}", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
    os._exit(0)  # don't wait for grader processes that hung
