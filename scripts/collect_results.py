"""README step 10: print the score tables for every evaluated model, and optionally save them as a CSV.

Each model's eval writes eval/outputs/<label>/log_<benchmark>.txt (label = the OUTPUT_DIR folder name). Usage:
    python collect_results.py [eval/outputs folder] [--csv ../results/scores.csv]
pass@8 is the share of questions where at least one of the 8 answers is right (eval.py prints it as "Acc:"), and
pass@1 is the share of right answers, both as the 4 decimal values in the log. Avg is the plain mean over all 8
benchmarks, shown once all 8 are done. The course leaderboard averages differently (it uses AIME 2025, which isn't in
the course's eval files, and leaves out MATH and AIME 2024), so our Avg can't be compared directly with the
leaderboard's.
"""
import argparse
import csv
import re
from pathlib import Path

BENCHES = ["aime", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"]


def read_scores(log):
    """(pass@8, pass@1, number of questions) from the last score lines of a log"""
    text = log.read_text(errors="ignore")
    acc = re.findall(r"^Acc: ([0-9.]+)", text, re.M)
    pass1 = re.findall(r"^Pass@1: .*= ([0-9.]+)", text, re.M)
    total = re.findall(r"^correct cnt / total cnt: \d+/(\d+)", text, re.M)
    return (float(acc[-1]) if acc else None, float(pass1[-1]) if pass1 else None, int(total[-1]) if total else None)


def print_table(title, results, which):
    print(f"\n{title}")
    print(f"{'model':20s}" + "".join(f"{b[:12]:>13s}" for b in BENCHES) + f"{'Avg':>9s}")
    for label, scores in results.items():
        vals = {b: scores[b][which] for b in BENCHES if b in scores and scores[b][which] is not None}
        cells = "".join(f"{vals[b]:13.4f}" if b in vals else f"{'-':>13s}" for b in BENCHES)
        avg = f"{sum(vals.values()) / len(BENCHES):9.4f}" if len(vals) == len(BENCHES) else f"{len(vals)}/8 done".rjust(9)
        print(f"{label:20s}{cells}{avg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent.parent / "eval" / "outputs")
    parser.add_argument("--csv", type=Path, help="also write model,benchmark,problems,pass_at_8,pass_at_1 here")
    args = parser.parse_args()

    results = {}
    for model_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        logs = {b: model_dir / f"log_{b}.txt" for b in BENCHES}
        scores = {b: read_scores(log) for b, log in logs.items() if log.exists()}
        if scores:
            results[model_dir.name] = scores
    if not results:
        raise SystemExit(f"no eval logs found under {args.root}")

    print_table("pass@8 (share of questions where at least one of the 8 answers is right)", results, 0)
    print_table("pass@1 (share of right answers)", results, 1)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "benchmark", "problems", "pass_at_8", "pass_at_1"])
            for label, scores in results.items():
                for b in BENCHES:
                    if b in scores:
                        w.writerow([label, b, scores[b][2], scores[b][0], scores[b][1]])
        print(f"\nsaved -> {args.csv}")
