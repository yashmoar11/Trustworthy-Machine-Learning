r"""README step 5, part 3: embed every math prompt, group the prompts by topic and find near duplicate questions.

The embeddings come from the sentence-transformers model all-MiniLM-L6-v2 (downloaded from the Hugging Face Hub).
Writes eval/train_data/clusters.parquet (row, c50, c300, nn_sim) and math_embeddings.npy (same row order):
  c50 / c300  topic cluster ids from k-means with 50 / 300 clusters (c50 to look at topics, c300 to sample from)
  nn_sim      cosine similarity to the most similar *other* math prompt (select_subsets.py treats 0.95 or more as a
              near duplicate)
It also prints how close each benchmark's questions are to the nearest math prompt in the training data. That's only
a coverage check, and nothing from it is used to pick training rows.

CPU only. Run from scripts/ with the select env (README step 1):
    sbatch -A tml_2026 -p normal_q --cpus-per-task=32 --mem=64G --time=1:00:00 --output=logs/cluster_%j.out \
        --wrap="$HOME/envs/select/bin/python cluster_prompts.py"
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval" / "train_data" / "acereason11_100k" / "data"
OUT_DIR = REPO / "eval" / "train_data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))


def nearest_other(emb, queries, chunk=4096, exclude_self=True):
    """Highest cosine similarity of each query to any row of emb (all embeddings are unit length). With exclude_self,
    the queries are emb itself and each row's match with itself is skipped."""
    best = np.empty(len(queries), dtype=np.float32)
    for s in range(0, len(queries), chunk):
        sims = queries[s : s + chunk] @ emb.T
        if exclude_self:
            sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -1
        best[s : s + chunk] = sims.max(axis=1)
    return best


if __name__ == "__main__":
    shards = sorted(SRC.glob("train-*.parquet"))
    rows = pd.concat((pq.read_table(p, columns=["instruction", "extra_info"]).to_pandas() for p in shards),
                     ignore_index=True)
    rows["row"] = range(len(rows))
    rows = rows[rows.extra_info.map(lambda e: e["category"]) == "math"].reset_index(drop=True)
    texts = rows.instruction.tolist()
    print(f"embedding {len(texts):,} math prompts with {MODEL} ...", flush=True)

    model = SentenceTransformer(MODEL, device="cpu")
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float32)
    np.save(OUT_DIR / "math_embeddings.npy", emb)

    out = pd.DataFrame({"row": rows.row.values})
    for k in (50, 300):
        out[f"c{k}"] = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb)
        print(f"k-means with {k} clusters done", flush=True)
    out["nn_sim"] = nearest_other(emb, emb)
    out.to_parquet(OUT_DIR / "clusters.parquet")
    print(f"saved -> {OUT_DIR / 'clusters.parquet'}", flush=True)

    print("\n=== near-duplicate questions (cosine similarity to the closest other math prompt) ===")
    for t in (0.90, 0.95, 0.98):
        print(f"  >= {t}: {(out.nn_sim >= t).mean():.1%} of math rows")

    print("\n=== 50 topic clusters: size, top words, two example prompts ===")
    tfidf = TfidfVectorizer(max_features=20000, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                            sublinear_tf=True, min_df=5)
    X = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())
    for c in out.c50.value_counts().index:
        idx = np.where(out.c50.values == c)[0]
        top = vocab[np.asarray(X[idx].mean(axis=0)).ravel().argsort()[::-1][:8]]
        print(f"\n[c50={c}] {len(idx):,} rows | {', '.join(top)}")
        for i in idx[:2]:
            print("    -", " ".join(texts[i].split())[:150])

    print("\n=== coverage check: similarity of each benchmark question to its nearest math training prompt ===")
    for bench_dir in sorted(p for p in (REPO / "eval" / "data").iterdir() if p.is_dir()):
        qs = []
        for line in filter(None, (bench_dir / "test.jsonl").read_text().split("\n")):
            row = json.loads(line)
            qs.append(next((row[k] for k in ("question", "problem", "Question", "input") if k in row), ""))
        q_emb = model.encode(qs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = nearest_other(emb, q_emb, exclude_self=False)
        print(f"  {bench_dir.name:14s} n={len(qs):4d}  median nearest-neighbour sim {np.median(sims):.2f}  "
              f"(share with no training prompt at 0.6 or more: {(sims < 0.6).mean():.0%})")
