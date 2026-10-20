r"""README steps 5 and 7: build the four 15K training subsets that start from a cleaned pool.

All four draw from the same cleaned pool of math rows. A row stays in the pool only if:
  - it's math (none of the benchmarks test code)
  - the whole example fits in 16,384 tokens, the limit we train with (longer ones get cut and lose the answer)
  - the answer written after the reasoning has exactly one \boxed{} (the eval only reads the last box)
  - the reasoning doesn't end by giving up ("given the time I've spent, I'll guess") or by saying the problem is
    broken or the answer is likely wrong (checked in the last 3,000 characters)
  - its final answer doesn't disagree with the source dataset's reference answer when both are plain numbers or
    option letters (the only comparison we trust)
  - its reasoning isn't in the 1% most repetitive of all math rows (measured by how much zlib can compress it)
  - its prompt shares no 50 character span with any benchmark question
  - it's the only solution kept for its question (same prompt text, ignoring case and spacing): the shortest verified
    correct one if there is one, else the shortest

The four subsets:
  cleaned_random
      uniform random draw from the pool
  cleaned_difficulty_weighted
      weighted random draw where the weight depends on the base model's solve rate: 1.25 under 0.125, 1.5 for 0.125
      to 0.625, 0.75 for 0.625 to 0.875, 0.25 for 0.875 and up (with 4 tries that's 0/4, 1/4 or 2/4, 3/4, 4/4)
  cleaned_difficulty_topic_balanced
      same weights, and all multiple choice problems go in first, the rest is spread over 300 topic clusters (quota
      grows with the square root of cluster size), and a question is skipped when its embedding (from
      cluster_prompts.py) has cosine similarity 0.95 or more with one already picked
  cleaned_hardest_only
      the problems with the lowest solve rate (longer solution first on ties), near duplicates skipped
A problem's solve rate is the share of its 4 tries the base model got right (measure_difficulty.py). A problem without
one (its final answer isn't a plain number or letter, or it was in shard 3, see DIFFICULTY_SHARDS) gets the mean
solve rate of the measured problems in its length decile (length of the whole example in tokens, mostly the solution).

Reads eval/train_data/merged.parquet (analysis_report.py), clusters.parquet and math_embeddings.npy
(cluster_prompts.py), difficulty_<shard>.parquet (measure_difficulty.py), the dataset's prompts and solutions, and
subsets/random_15k_rows.json (README step 4, for the comparison table at the end).
Run from scripts/ with the select env (README step 1):
    $HOME/envs/select/bin/python select_subsets.py --candidates   # problem list for measure_difficulty.py
    $HOME/envs/select/bin/python select_subsets.py                # subsets/<name>_rows.json for all four
Then make_subset.py turns each row list into a training file.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "eval" / "train_data"
OUT_DIR = REPO / "subsets"
# Subset i in this list uses seed --seed + i, so changing the order changes the rows.
SUBSETS = ["cleaned_random", "cleaned_difficulty_weighted", "cleaned_difficulty_topic_balanced", "cleaned_hardest_only"]
# Our subsets were built without shard 3 of measure_difficulty.py (its job ran out of time while grading), so its
# problems got the length based estimate. Leave 3 out to get our exact rows, or add it back to use every measurement.
DIFFICULTY_SHARDS = [0, 1, 2, 4, 5]

FLAWED = re.compile(
    r"ill-posed|ill posed|(?:there(?:'s| is| must be) (?:a |an )?)(?:typo|mistake|error) in the problem|"
    r"problem (?:is|seems|might be) (?:flawed|incorrect|wrong|missing)|missing information|"
    r"(?:likely|probably) incorrect|I know this is (?:wrong|incorrect)|cannot be (?:solved|determined) "
    r"(?:as stated|with the given)|not enough information",
    re.I,
)
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
LETTER = re.compile(r"^[A-E]$")


def plain(ans):
    """The answer as a plain number or option letter, else None (we only trust those comparisons)."""
    if ans is None:
        return None
    a = re.sub(r"\\[,!; ]|\\text(?:bf)?\{([^}]*)\}|\^\\circ|\\%|[{}\s,$]", r"\1", ans).strip("()")
    return a if NUMBER.match(a) or LETTER.match(a) else None


def load_texts(rows, column):
    shards = sorted((DATA / "acereason11_100k" / "data").glob("train-*.parquet"))
    col = pd.concat((pq.read_table(p, columns=[column]).to_pandas() for p in shards), ignore_index=True)[column]
    return col.iloc[rows].values


def filtered_pool(df):
    m = df[df.category == "math"].copy()
    m["mcq"] = m.mcq.fillna(False).astype(bool)  # it's an object column after the merge
    m["flawed"] = [bool(FLAWED.search(o.partition("</think>")[0][-3000:])) for o in load_texts(m.row.values, "output")]
    m["p_pred"], m["p_ref"] = m.pred_answer.map(plain), m.gt_answer.map(plain)
    m["verified_wrong"] = m.p_pred.notna() & m.p_ref.notna() & (m.correct == False)  # noqa: E712
    loop_cut = m.think_zlib_ratio.quantile(0.01)
    steps = [
        ("math rows", lambda d: d.category == "math"),
        ("fits in 16,384 tokens", lambda d: ~d.truncated),
        ("exactly one \\boxed in final answer", lambda d: d.n_boxed_answer == 1),
        ("no give-up ending", lambda d: ~d.guess_end),
        ("no 'problem is broken / likely wrong'", lambda d: ~d.flawed),
        ("doesn't disagree with the reference answer", lambda d: ~d.verified_wrong),
        ("not in the 1% most repetitive", lambda d: d.think_zlib_ratio >= loop_cut),
        ("no benchmark overlap", lambda d: d.contam_bench == ""),
    ]
    print("cleaned pool:")
    for name, cond in steps:
        m = m[cond(m)]
        print(f"  {name:40s} {len(m):7,} rows")
    m["verified_correct"] = m.p_pred.notna() & m.p_ref.notna() & (m.correct == True)  # noqa: E712
    m = m.sort_values(["verified_correct", "total_tok"], ascending=[False, True]).drop_duplicates("prompt_hash")
    print(f"  {'one solution per question':40s} {len(m):7,} rows")
    return m.reset_index(drop=True)


def add_difficulty(pool):
    """Adds solve_est (the base model's solve rate, measured where we have it, else estimated from length) and the
    sampling weight that goes with it."""
    files = [f for f in (DATA / f"difficulty_{i}.parquet" for i in DIFFICULTY_SHARDS) if f.exists()]
    if files:
        st = pd.concat(pd.read_parquet(f) for f in files)
        pool = pool.merge(st[["row", "n_correct", "n_samples"]], on="row", how="left")
        pool["solve_rate"] = pool.n_correct / pool.n_samples
    else:
        pool["solve_rate"] = np.nan
    measured = pool[pool.solve_rate.notna()]
    if len(measured) >= 1000:
        edges = np.unique(np.quantile(measured.total_tok, np.linspace(0, 1, 11)))
        by_len = measured.groupby(pd.cut(measured.total_tok, edges, include_lowest=True), observed=False).solve_rate.mean()
        predicted = pd.cut(pool.total_tok.clip(edges[0], edges[-1]), edges, include_lowest=True).map(by_len).astype(float)
        pool["solve_est"] = pool.solve_rate.fillna(predicted)
        print(f"\ndifficulty: base-model solve rate measured on {len(measured):,} problems, predicted from length "
              f"for {pool.solve_rate.isna().sum():,}")
        print("  measured solve rate:", measured.solve_rate.value_counts().sort_index().to_dict())
        print("  mean solve rate by length decile:", by_len.round(2).tolist())
        pool["weight"] = np.select([pool.solve_est >= 0.875, pool.solve_est >= 0.625, pool.solve_est >= 0.125],
                                   [0.25, 0.75, 1.5], 1.25)
    else:
        # fewer than 1,000 measured problems: use solution length alone (longer = harder)
        print("\ndifficulty: fewer than 1,000 base-model measurements, using solution length instead (longer = harder)")
        pool["solve_est"] = 1 - pool.total_tok.rank(pct=True)
        pool["weight"] = np.select([pool.solve_est <= 0.3, pool.solve_est <= 0.7], [1.5, 1.0], 0.5)
    return pool


class Picker:
    """Collects up to n rows. With dup_sim set, skips a question whose embedding has cosine similarity dup_sim or
    more with one already picked."""

    def __init__(self, n, emb, emb_idx, dup_sim=None):
        self.n, self.emb, self.emb_idx, self.dup_sim = n, emb, emb_idx, dup_sim
        self.rows, self.mat = [], np.zeros((n, emb.shape[1]), dtype=np.float32)

    def add(self, r):
        if len(self.rows) == self.n:
            return False
        v = self.emb[self.emb_idx[r]]
        if self.dup_sim and self.rows and (self.mat[: len(self.rows)] @ v).max() >= self.dup_sim:
            return False
        self.mat[len(self.rows)] = v
        self.rows.append(r)
        return True


def weighted_order(weights, rng):
    """Random order where items with bigger weights tend to come first, so the first n are a weighted sample
    without replacement (Efraimidis-Spirakis: sort by u ** (1 / weight), with u uniform in [0, 1))."""
    return np.argsort(-(rng.random(len(weights)) ** (1.0 / np.asarray(weights, dtype=float))))


def select(name, pool, emb, emb_idx, n, rng):
    if name == "cleaned_random":
        return list(pool.row.values[rng.permutation(len(pool))[:n]])
    if name == "cleaned_difficulty_weighted":
        return list(pool.row.values[weighted_order(pool.weight.values, rng)[:n]])
    if name == "cleaned_hardest_only":
        picker = Picker(n, emb, emb_idx, dup_sim=0.95)
        for r in pool.sort_values(["solve_est", "total_tok"], ascending=[True, False]).row.values:
            if len(picker.rows) == n:
                break
            picker.add(r)
        return picker.rows
    # cleaned_difficulty_topic_balanced: every multiple choice problem first, then quotas over the 300 topic clusters
    picker = Picker(n, emb, emb_idx, dup_sim=0.95)
    mcq = pool[pool.mcq]
    for r in mcq.row.values[weighted_order(mcq.weight.values, rng)]:
        picker.add(r)
    rest = pool[~pool.mcq]
    sizes = rest.groupby("c300").size()
    target = n - len(picker.rows)
    quota = np.minimum(np.floor(np.sqrt(sizes) / np.sqrt(sizes).sum() * target), sizes).astype(int)
    while quota.sum() < target:  # hand out the rounding leftovers to random clusters that still have room
        room = sizes - quota
        quota[room[room > 0].index[rng.integers((room > 0).sum())]] += 1
    for c, q in quota.items():
        grp = rest[rest.c300 == c]
        taken = 0
        for r in grp.row.values[weighted_order(grp.weight.values, rng)]:
            if taken == q:
                break
            taken += picker.add(r)
    left = pool[~pool.row.isin(set(picker.rows))]  # fill the slots left by skipped near duplicates
    for r in left.row.values[weighted_order(left.weight.values, rng)]:
        if len(picker.rows) == n:
            break
        picker.add(r)
    return picker.rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", action="store_true", help="write the problem list for measure_difficulty.py")
    args = parser.parse_args()

    df = pd.read_parquet(DATA / "merged.parquet")
    pool = filtered_pool(df)

    if args.candidates:
        cand = pool[pool.p_pred.notna()][["row", "p_pred"]].rename(columns={"p_pred": "teacher_answer"})
        cand["instruction"] = load_texts(cand.row.values, "instruction")
        cand.to_parquet(DATA / "difficulty_candidates.parquet")
        print(f"\n{len(cand):,} candidates whose solution's final answer is a plain number or letter "
              f"-> {DATA / 'difficulty_candidates.parquet'}")
        raise SystemExit

    pool = add_difficulty(pool)
    clus = pd.read_parquet(DATA / "clusters.parquet")  # same row order as math_embeddings.npy
    emb = np.load(DATA / "math_embeddings.npy")
    emb_idx = pd.Series(np.arange(len(clus)), index=clus.row.values)

    by_row = df.set_index("row")
    summary = {"random_15k": by_row.loc[json.loads((OUT_DIR / "random_15k_rows.json").read_text())]}
    for i, name in enumerate(SUBSETS):
        rows = sorted(int(r) for r in select(name, pool, emb, emb_idx, args.n, np.random.default_rng(args.seed + i)))
        assert len(rows) == args.n == len(set(rows))
        (OUT_DIR / f"{name}_rows.json").write_text(json.dumps(rows))
        summary[name] = by_row.loc[rows].join(pool.set_index("row")[["solve_est"]])

    def describe(d):
        return pd.Series({
            "math": f"{(d.category == 'math').mean():.0%}",
            "mean_tokens": int(d.total_tok.clip(upper=16384).mean()),
            "est_base_solve_rate": f"{d.solve_est.mean():.2f}" if "solve_est" in d else "-",
            "give_up_or_cut": f"{(d.guess_end | d.truncated).mean():.1%}",
            "mcq": f"{d.mcq.fillna(False).astype(bool).mean():.1%}",
            "topics(c300)": d.c300.nunique(),
            "repeat_questions": int(d.prompt_hash.duplicated().sum()),
        })

    print("\n" + pd.DataFrame({k: describe(v) for k, v in summary.items()}).to_string())
    print(f"\nwrote {', '.join(f'{name}_rows.json' for name in SUBSETS)} to {OUT_DIR}")
