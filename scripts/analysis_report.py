r"""README step 5, part 4: join everything per row and print the tables we used to design the selection.

Reads features.parquet (analyze_data.py), linked.parquet (link_and_grade.py) and clusters.parquet (cluster_prompts.py)
from eval/train_data/, plus the dataset's prompts, the benchmark questions in eval/data/ and
subsets/random_15k_rows.json (README step 4). Writes eval/train_data/merged.parquet, which select_subsets.py
reads, so run this before it.

Only reads small tables, so a login node is fine. Run from scripts/ with the select env (README step 1):
    $HOME/envs/select/bin/python analysis_report.py
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "eval" / "train_data"
SHINGLE = 50  # same 50 character span as the overlap check in analyze_data.py
STRIP = re.compile(r"[\s$\\{}]")
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def pct(x):
    return f"{100 * x:.1f}%" if pd.notna(x) else "-"


def plain_number(ans):
    if ans is None:
        return None
    a = re.sub(r"\\[,!; ]|\\text\{[^}]*\}|\^\\circ|\\%|[{}\s,$]", "", ans)
    return a if NUMBER.match(a) else None


def summarise(g):
    graded = g[g.correct.notna()]
    return pd.Series({
        "rows": len(g),
        "med_tok": int(g.total_tok.median()),
        "med_pass72b": g.pass_rate_72b.median(),
        "graded": len(graded),
        "correct": pct(graded.correct.astype(float).mean()) if len(graded) else "-",
        "guess_end": pct(g.guess_end.mean()),
        "trunc": pct(g.truncated.mean()),
        "mcq": pct(g.mcq.mean()),
    })


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


if __name__ == "__main__":
    feats = pd.read_parquet(DATA / "features.parquet")
    enr = pd.read_parquet(DATA / "linked.parquet").drop(columns="source")
    clus = pd.read_parquet(DATA / "clusters.parquet")
    df = feats.merge(enr, on="row", how="left").merge(clus, on="row", how="left")

    shards = sorted((DATA / "acereason11_100k" / "data").glob("train-*.parquet"))
    prompts = pd.concat((pq.read_table(p, columns=["instruction"]).to_pandas() for p in shards), ignore_index=True)
    check = prompts.instruction.map(lambda p: hashlib.md5(" ".join(p.lower().split()).encode()).hexdigest())
    assert (check.values == feats.sort_values("row").prompt_hash.values).all(), "row numbering mismatch"

    m = df[df.category == "math"].copy()
    m["pred_num"], m["gt_num"] = m.pred_answer.map(plain_number), m.gt_answer.map(plain_number)
    m["numeric_pair"] = m.pred_num.notna() & m.gt_num.notna()
    m["pass_bin"] = pd.cut(m.pass_rate_72b, [-0.01, 0.0, 0.25, 0.5, 0.75, 0.99, 1.0],
                           labels=["0 (72B never)", "0-25%", "25-50%", "50-75%", "75-99%", "100% (72B always)"])
    m["tok_bin"] = pd.cut(m.total_tok, [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16384, 10**7],
                          labels=["<2K", "2-4K", "4-6K", "6-8K", "8-10K", "10-12K", "12-14K", "14-16.4K", ">16.4K"])

    section("1. Linking AceReason math rows to their source problems")
    print(f"math rows: {len(m):,}")
    m["found"] = m.omr_problem_source.notna() | m.numina_source.notna()
    print(m.groupby("source").apply(lambda g: pd.Series({
        "rows": len(g), "found_in_source": pct(g.found.mean()), "has_ref_answer": pct(g.gt_answer.notna().mean()),
        "gradable": pct(g.correct.notna().mean()), "has_pass_rate": pct(g.pass_rate_72b.notna().mean())}),
        include_groups=False).to_string())

    section("2. Is the solution's final answer correct? (gradable rows only)")
    g = m[m.correct.notna()]
    print(f"gradable rows: {len(g):,} ({pct(len(g) / len(m))} of math)")
    print(f"  correct overall: {pct(g.correct.astype(float).mean())}")
    num = g[g.numeric_pair]
    print(f"  both answers plain numbers (most reliable comparison): {len(num):,} rows, "
          f"correct {pct(num.correct.astype(float).mean())}")
    for col in ("source", "omr_problem_type", "guess_end", "tok_bin", "pass_bin"):
        t = g.groupby(col, observed=True).correct.agg(["count", lambda s: pct(s.astype(float).mean())])
        t.columns = ["graded", "correct"]
        print(f"\n  by {col}:\n" + t.to_string())

    section("3. Difficulty: pass rate of Qwen2.5-72B (with tools) from OpenMathReasoning (OMR), 32 tries per problem")
    o = m[m.pass_rate_72b.notna()]
    print(f"rows with a pass rate: {len(o):,}")
    print(o.groupby("pass_bin", observed=True).apply(summarise, include_groups=False).to_string())
    print(f"\nSpearman correlation with pass rate: length {o.pass_rate_72b.corr(o.total_tok, method='spearman'):.2f}, "
          f"reflection words (Wait, Hmm, ...) {o.pass_rate_72b.corr(o.n_reflect, method='spearman'):.2f}")

    section("4. Where the problems come from")
    print("OpenMathReasoning problem_source:")
    print(m[m.omr_problem_source.notna()].groupby("omr_problem_source").apply(summarise, include_groups=False)
          .sort_values("rows", ascending=False).to_string())
    print("\nNuminaMath-CoT source:")
    print(m[m.numina_source.notna()].groupby("numina_source").apply(summarise, include_groups=False)
          .sort_values("rows", ascending=False).to_string())

    section("5. Answer format and text quality")
    print(f"multiple-choice prompts: {pct(m.mcq.mean())} of math ({m.mcq.sum():,} rows); "
          f"by source: {m.groupby('source').mcq.mean().map(pct).to_dict()}")
    letter = m.pred_answer.fillna("").str.fullmatch(r"\(?(?:\\text\{)?\(?[A-E]\)?\}?\)?")
    print(f"solution's final answer is a bare option letter: {letter.sum():,} rows")
    print(f"more than one \\boxed in the final answer: {pct((m.n_boxed_answer > 1).mean())} "
          f"(the eval only reads the last one)")
    print(f"Chinese characters anywhere in the output: {pct((m.cjk_output > 0).mean())}")
    q = m.think_zlib_ratio.quantile([0.01, 0.05, 0.5]).round(3).to_dict()
    low = m[m.think_zlib_ratio < m.think_zlib_ratio.quantile(0.01)]
    print(f"reasoning compressibility (zlib ratio; low = repetitive): quantiles {q}")
    print(f"  most repetitive 1%: give-up ending {pct(low.guess_end.mean())}, truncated {pct(low.truncated.mean())}, "
          f"correct {pct(low.correct.dropna().astype(float).mean())}")

    section("6. Duplicates")
    print(f"exact duplicate prompts: {pct((m.dup_count > 1).mean())} of math rows share their prompt with another row")
    for t in (0.9, 0.95, 0.98):
        print(f"  near-duplicates (embedding similarity >= {t}): {pct((m.nn_sim >= t).mean())}")

    section("7. The 50 topic clusters (c50); their top words and example prompts are in cluster_prompts.py's log")
    print(m.groupby("c50").apply(summarise, include_groups=False).sort_values("rows", ascending=False).to_string())

    section("8. Overlap with benchmark questions: how much of the question each flagged row contains")
    bench = {}
    for d in sorted(p for p in (REPO / "eval" / "data").iterdir() if p.is_dir()):
        for line in filter(None, (d / "test.jsonl").read_text().split("\n")):
            r = json.loads(line)
            qn = STRIP.sub("", next((r[k] for k in ("question", "problem", "Question", "input") if k in r), "").lower())
            bench.setdefault(d.name, []).append({qn[i : i + SHINGLE] for i in range(max(1, len(qn) - SHINGLE + 1))})
    cover = []
    for r in df[df.contam_bench != ""].itertuples():
        text = STRIP.sub("", prompts.instruction[r.row].lower())
        shingles = {text[i : i + SHINGLE] for i in range(max(1, len(text) - SHINGLE + 1))}
        best = max((len(s & shingles) / len(s), name) for name, qs in bench.items() for s in qs)
        cover.append((r.row, best[1], best[0]))
    cov = pd.DataFrame(cover, columns=["row", "bench", "coverage"])
    df = df.merge(cov[["row", "coverage"]].rename(columns={"coverage": "contam_coverage"}), on="row", how="left")
    print("flagged rows by share of the benchmark question they contain:")
    print(cov.groupby("bench").coverage.describe()[["count", "min", "50%", "max"]].to_string())
    print(f"rows containing >= 50% of a benchmark question: {(cov.coverage >= 0.5).sum()} "
          f"(by benchmark: {cov[cov.coverage >= 0.5].bench.value_counts().to_dict()})")
    for r in cov.sort_values("coverage").head(3).itertuples():
        print(f"  weakest match ({r.coverage:.0%} of a {r.bench} question): {prompts.instruction[r.row][:140]!r}")

    section("9. What the random 15K baseline actually contains")
    rnd = df[df.row.isin(json.loads((REPO / "subsets" / "random_15k_rows.json").read_text()))]
    rm = rnd[rnd.category == "math"]
    print(f"rows: {len(rnd):,} | code: {pct((rnd.category == 'code').mean())} | truncated: {pct(rnd.truncated.mean())} "
          f"| give-up ending: {pct(rnd.guess_end.mean())} | mean tokens (counting at most 16,384): {rnd.total_tok.clip(upper=16384).mean():.0f}")
    wrong = rm[rm.correct == False]  # noqa: E712
    print(f"math rows graded wrong: {len(wrong):,} of {rm.correct.notna().sum():,} gradable "
          f"| overlap >= 50% with a benchmark question: {(rnd.contam_coverage >= 0.5).sum()} "
          f"| repeated prompts inside the subset: {rnd.prompt_hash.duplicated().sum()}")

    df.to_parquet(DATA / "merged.parquet")
    print(f"\nsaved -> {DATA / 'merged.parquet'}")
