"""README step 5, part 1: record a few facts about every row of the training data before picking anything.

The training data is the course's acereason11_100k dataset: 100,000 rows, each a math or coding problem with a long
step by step solution (the reasoning, then a final answer after a </think> tag). For every row this records:
  - token counts, and whether the whole example fits in 16,384 tokens, the limit we train with (cutoff_len)
  - whether the part after </think> has a boxed answer or a code block
  - how often the reasoning says things like "Wait" or "Let me check"
  - whether the reasoning ends by giving up
  - whether the prompt has Chinese characters
  - a hash of the prompt, to find repeated questions
  - whether the prompt shares a 50 character span with any benchmark question
It reads the dataset's parquet files, the base model's tokenizer and the benchmark questions in eval/data/, prints
summary tables, and saves the per-row facts to eval/train_data/features.parquet for the next steps.

CPU only. Run from scripts/ with the sft env (README step 1):
    sbatch -A tml_2026 -p normal_q --cpus-per-task=16 --mem=64G --time=1:00:00 \
        --output=logs/analyze_%j.out --wrap="$HOME/envs/sft/bin/python analyze_data.py"
"""
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval" / "train_data" / "acereason11_100k" / "data"
BENCH_DIR = REPO / "eval" / "data"
TOKENIZER = REPO / "eval" / "models" / "Qwen2.5-3B-Instruct"
OUT = REPO / "eval" / "train_data" / "features.parquet"

CUTOFF = 16384
TEMPLATE_TOKENS = 15  # chat template tokens around the system, user and assistant turns
SHINGLE = 50  # a shared 50 character span (after normalizing) counts as overlap with an eval question
NUM_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", min(16, os.cpu_count() or 1)))

REFLECT = re.compile(r"\b(?:[Ww]ait|Alternatively|Hmm|[Dd]ouble-check|[Ll]et me (?:check|verify))\b")
# A solution giving up in the last 3,000 characters of its reasoning, like "given that I'm stuck, I'll hazard a guess".
# Strong phrases only, since softer ones like "perhaps the answer is" also show up in normal solutions.
GUESS = re.compile(
    r"hazard a guess|(?:educated|best) guess|I(?:'ll| will| have to| must) (?:just )?guess|I'm stuck|I am stuck|"
    r"(?:due to|given) (?:the )?time constraints|given the time I've spent|I(?:'ll| will) proceed with the answer|"
    r"I must have made a (?:mistake|wrong)|I can't (?:figure|solve|find)|unable to (?:solve|find|determine)",
    re.I,
)
CJK = re.compile(r"[一-鿿]")
STRIP = re.compile(r"[\s$\\{}]")


def normalise(text):
    return STRIP.sub("", text.lower())


def load_benchmark_shingles():
    shingles = {}
    for bench_dir in sorted(p for p in BENCH_DIR.iterdir() if p.is_dir()):
        # split on "\n" only, since splitlines() also splits on unicode separators that appear inside some questions
        for line in filter(None, (bench_dir / "test.jsonl").read_text().split("\n")):
            row = json.loads(line)
            question = next((row[k] for k in ("question", "problem", "Question", "input") if k in row), "")
            text = normalise(question)
            for i in range(max(1, len(text) - SHINGLE + 1)):
                shingles.setdefault(text[i : i + SHINGLE], bench_dir.name)
    return shingles


BENCH_SHINGLES = load_benchmark_shingles()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def featurise(batch, indices):
    prompt_tok = [len(x) for x in tokenizer(batch["instruction"], add_special_tokens=False)["input_ids"]]
    out_tok = [len(x) for x in tokenizer(batch["output"], add_special_tokens=False)["input_ids"]]
    sys_tok = [len(x) for x in tokenizer(batch["system"], add_special_tokens=False)["input_ids"]]
    feats = {k: [] for k in (
        "row", "category", "source", "prompt_tok", "out_tok", "total_tok", "truncated", "has_think_end",
        "think_frac", "boxed_in_answer", "code_in_answer", "n_reflect", "guess_end", "cjk_prompt", "prompt_hash",
        "contam_bench",
    )}
    for i, (row, info, prompt, output) in enumerate(zip(indices, batch["extra_info"], batch["instruction"], batch["output"])):
        think, sep, answer = output.partition("</think>")
        total = prompt_tok[i] + out_tok[i] + sys_tok[i] + TEMPLATE_TOKENS
        norm = normalise(prompt)
        hit = ""
        for j in range(max(1, len(norm) - SHINGLE + 1)):
            hit = BENCH_SHINGLES.get(norm[j : j + SHINGLE], "")
            if hit:
                break
        feats["row"].append(row)
        feats["category"].append(info["category"])
        feats["source"].append(info["source"])
        feats["prompt_tok"].append(prompt_tok[i])
        feats["out_tok"].append(out_tok[i])
        feats["total_tok"].append(total)
        feats["truncated"].append(total > CUTOFF)
        feats["has_think_end"].append(bool(sep))
        feats["think_frac"].append(len(think) / max(len(output), 1))
        feats["boxed_in_answer"].append("\\boxed" in answer)
        feats["code_in_answer"].append("```" in answer)
        feats["n_reflect"].append(len(REFLECT.findall(think)))
        feats["guess_end"].append(bool(GUESS.search(think[-3000:])))
        feats["cjk_prompt"].append(bool(CJK.search(prompt)))
        feats["prompt_hash"].append(hashlib.md5(" ".join(prompt.lower().split()).encode()).hexdigest())
        feats["contam_bench"].append(hit)
    return feats


def pct(x):
    return f"{100 * x:.1f}%"


if __name__ == "__main__":
    ds = load_dataset("parquet", data_files=str(SRC / "train-*.parquet"), split="train")
    feats = ds.map(featurise, batched=True, batch_size=200, with_indices=True, num_proc=NUM_PROC,
                   remove_columns=ds.column_names, desc="featurising")
    df = feats.to_pandas()
    df["dup_count"] = df.groupby("prompt_hash")["row"].transform("size")
    df.to_parquet(OUT)
    print(f"\nsaved per-row features -> {OUT}")

    print("\n=== 1. Overview ===")
    print(f"rows: {len(df):,}   unique prompts: {df.prompt_hash.nunique():,}")
    print(df.category.value_counts().to_string())

    print("\n=== 2. By category and source ===")
    g = df.groupby(["category", "source"])
    table = pd.DataFrame({
        "rows": g.size(),
        "uniq_prompts": g.prompt_hash.nunique(),
        "med_tok": g.total_tok.median().astype(int),
        "p90_tok": g.total_tok.quantile(0.9).astype(int),
        "trunc": g.truncated.mean().map(pct),
        "no_think_end": (1 - g.has_think_end.mean()).map(pct),
        "boxed": g.boxed_in_answer.mean().map(pct),
        "med_reflect": g.n_reflect.median(),
        "guess_end": g.guess_end.mean().map(pct),
        "chinese": g.cjk_prompt.mean().map(pct),
    }).sort_values("rows", ascending=False)
    print(table.to_string())

    print("\n=== 3. Length of full training sequence (tokens), math vs code ===")
    bins = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16384, 10**7]
    labels = ["<2K", "2-4K", "4-6K", "6-8K", "8-10K", "10-12K", "12-14K", "14-16.4K", ">16.4K (cut)"]
    df["len_bin"] = pd.cut(df.total_tok, bins=bins, labels=labels, right=False)
    print(pd.crosstab(df.len_bin, df.category).to_string())

    print("\n=== 4. Responses per prompt (same question appearing several times) ===")
    per_prompt = df.groupby("prompt_hash").size()
    print(per_prompt.value_counts().sort_index().head(15).to_string())

    print("\n=== 5. Math quality checks ===")
    m = df[df.category == "math"]
    print(f"math rows: {len(m):,}")
    print(f"  truncated at {CUTOFF}: {m.truncated.sum():,} ({pct(m.truncated.mean())})")
    print(f"  missing </think>: {(~m.has_think_end).sum():,}")
    print(f"  no \\boxed in final answer: {(~m.boxed_in_answer).sum():,}")
    print(f"  prompts with Chinese characters: {m.cjk_prompt.sum():,}")
    print("  reflection markers per solution (Wait/Hmm/Alternatively/check), quantiles:")
    print("   ", m.n_reflect.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
    print("  correlation of length with reflection count:", round(m.total_tok.corr(m.n_reflect), 3))
    print("  share ending in a guess / giving up, by length bin:")
    print("   ", m.groupby("len_bin", observed=True).guess_end.mean().map(pct).to_dict())

    print("\n=== 6. Overlap with eval benchmarks (shared 50-char span) ===")
    hits = df[df.contam_bench != ""]
    print(f"rows overlapping a benchmark question: {len(hits):,}")
    print(hits.groupby(["contam_bench", "category"]).size().to_string() if len(hits) else "(none)")
    for r in hits.head(8).itertuples():
        print(f"  row {r.row} [{r.contam_bench}]: {ds[int(r.row)]['instruction'][:160]!r}")
