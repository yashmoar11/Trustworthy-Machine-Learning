"""README steps 4 and 7: turn a list of row numbers into a LLaMA-Factory training file.

Row numbers are positions in the course's 100,000 row training dataset (acereason11_100k, as read from its parquet
files). Run from scripts/ with the sft env (README step 1).

Random baseline (15,000 rows picked uniformly at random from all 100,000, seed 42):
    $HOME/envs/sft/bin/python make_subset.py --name random_15k --random 15000
From a row list written by select_subsets.py:
    $HOME/envs/sft/bin/python make_subset.py --name cleaned_difficulty_weighted --rows ../subsets/cleaned_difficulty_weighted_rows.json

Writes LLaMA-Factory/data/<name>.json (the training file), subsets/<name>_rows.json (the row numbers, enough to
rebuild the subset) and registers <name> in LLaMA-Factory/data/dataset_info.json.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent
LF_DATA = REPO / "LLaMA-Factory" / "data"
SUBSETS = REPO / "subsets"
DEFAULT_SRC = REPO / "eval" / "train_data" / "acereason11_100k" / "data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="dataset name used in the training config, e.g. random_15k")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--random", type=int, metavar="N", help="pick N rows uniformly at random")
    group.add_argument("--rows", type=Path, help="JSON file with a list of row numbers to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="folder with the dataset's train-*.parquet files")
    args = parser.parse_args()

    # The dataset's parquet files have the same 100,000 rows as acereason11_100k.json and load much faster.
    ds = load_dataset("parquet", data_files=str(args.src / "train-*.parquet"), split="train")
    assert len(ds) == 100_000, f"expected 100,000 rows, found {len(ds)}"

    if args.random:
        idx = sorted(random.Random(args.seed).sample(range(len(ds)), args.random))
    else:
        idx = sorted(set(json.loads(args.rows.read_text())))
    subset = ds.select(idx)

    rows = [
        {"instruction": r["instruction"], "input": r["input"], "output": r["output"], "system": r["system"]}
        for r in subset
    ]
    out_file = LF_DATA / f"{args.name}.json"
    out_file.write_text(json.dumps(rows, ensure_ascii=False))
    SUBSETS.mkdir(exist_ok=True)
    (SUBSETS / f"{args.name}_rows.json").write_text(json.dumps(idx))

    # Map the system column too. Without it LLaMA-Factory drops the dataset's system prompt and trains with Qwen's
    # default one, which isn't the prompt the eval uses.
    info_file = LF_DATA / "dataset_info.json"
    info = json.loads(info_file.read_text())
    info[args.name] = {
        "file_name": out_file.name,
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"},
    }
    info_file.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {out_file}")
    print("categories:", dict(collections.Counter(r["extra_info"]["category"] for r in subset)))
    print("system prompts:", dict(collections.Counter(r["system"] for r in rows)))
    print(f"registered '{args.name}' in {info_file}")
