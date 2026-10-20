# Picking 15K training rows for reasoning fine-tuning (Qwen2.5-3B-Instruct)

Class project for Trustworthy Machine Learning at Virginia Tech (Fall 2026). We ran everything on ARC, Virginia Tech's
Advanced Research Computing clusters. Jobs there go through Slurm, the job scheduler you submit work to with `sbatch`.

The assignment is to fully fine-tune [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on 15,000
training examples and score it on 8 math and science benchmarks with the course's eval. The examples have to come from
the course's training dataset, [acereason11_100k](https://huggingface.co/datasets/redsgnaoh/acereason11_100k): 100,000
rows taken from NVIDIA's [AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT). Each row is a
math or coding problem plus a long step by step solution that ends in a final answer. The whole project is about which
15,000 of those 100,000 rows to train on, and picking them at random is the baseline to beat.

Before picking anything, we analyzed the whole dataset row by row. That turned up a lot of rows that can't help on
these benchmarks or that are bad examples to learn from: coding problems (the benchmarks are all math and science),
solutions longer than the 16,384 token limit we train with, solutions that give up, final answers that disagree with
the source dataset's reference answer, solutions that repeat the same text over and over, questions that overlap with
the benchmarks, and repeated questions. We threw those out and built four training sets from the rest. We trained one
model on each of them plus one on a random pick of all 100,000 rows, five models in total. Every model is
Qwen2.5-3B-Instruct fully fine-tuned for 3 epochs on its 15,000 rows with the exact same settings, so the only
difference between them is the data, and each model is named after its training set:

- random_15k: 15,000 rows picked at random from all 100,000, math and code, nothing removed. This is the baseline
  the assignment asks for.
- cleaned_random: 15,000 rows picked at random from the cleaned pool (the math rows left after the cleaning).
- cleaned_difficulty_weighted: 15,000 rows from the cleaned pool, picked at random but weighted so that problems the
  base model (Qwen2.5-3B-Instruct before any fine-tuning) often got wrong in our test are picked more often. This is
  our final model.
- cleaned_difficulty_topic_balanced: the same weighting, plus every multiple choice problem, a spread over 300 topic
  groups, and no near duplicate questions.
- cleaned_hardest_only: the 15,000 problems in the cleaned pool that the base model solved least often in our test.

"How we picked the data" below has the details.

Our final model is **cleaned_difficulty_weighted**. Averaged over the 8 benchmarks, its pass@8 (the share of questions
where at least one of its 8 tries is right, see Results) is 0.582, compared with 0.535 for the base model and 0.548
for random_15k. Its average pass@1 (how often a single answer is right) is 0.289, compared with 0.285 and 0.272.

## Results

For every benchmark question, the model writes 8 separate answers. We report two scores:
- pass@8 is the share of questions where at least one of the 8 answers is right. The course leaderboard ranks by this
  one (its table says "Pass@k: k=8").
- pass@1 is the share of right answers out of all of them, so it's how often a single answer is right. The course's
  `eval/README.md` recommends reporting this one, so we show both.

Avg is the plain mean over the 8 benchmarks, so every benchmark counts the same no matter how many questions it has.
The number of questions is in parentheses in the pass@8 table's header. All values are also in
[results/scores.csv](results/scores.csv) with 4 decimals. The tables below are rounded from the exact counts in the
eval logs, so three cells differ by 0.001 from rounding the CSV's values. The benchmarks, as
the course README names them, are AIME 2024 and AMC 2023 (US math competitions), CN_MATH 2024 (Chinese math
competitions, 2024), GPQA Diamond (science, multiple choice), Kaoyan (the Chinese graduate entrance exam), MATH-500,
Minerva Math and OlympiadBench.

**pass@8**

| model | AIME 2024 (30) | AMC 2023 (40) | CN_MATH 2024 (30) | GPQA Diamond (198) | Kaoyan (199) | MATH-500 (500) | Minerva (272) | OlympiadBench (675) | Avg |
|---|---|---|---|---|---|---|---|---|---|
| base model, no fine-tuning | 0.233 | 0.700 | 0.267 | 0.737 | 0.518 | 0.846 | 0.482 | 0.496 | 0.535 |
| random_15k | 0.300 | 0.700 | 0.200 | 0.707 | 0.668 | 0.878 | 0.434 | 0.499 | 0.548 |
| cleaned_random | 0.200 | 0.675 | 0.367 | 0.717 | 0.693 | 0.880 | 0.430 | 0.554 | 0.565 |
| **cleaned_difficulty_weighted** (final) | 0.233 | 0.775 | 0.433 | 0.672 | 0.678 | 0.882 | 0.441 | 0.544 | **0.582** |
| cleaned_difficulty_topic_balanced | 0.233 | 0.775 | 0.300 | 0.687 | 0.688 | 0.890 | 0.430 | 0.557 | 0.570 |
| cleaned_hardest_only | 0.233 | 0.600 | 0.367 | 0.682 | 0.663 | 0.882 | 0.471 | 0.554 | 0.556 |

**pass@1**

| model | AIME 2024 | AMC 2023 | CN_MATH 2024 | GPQA Diamond | Kaoyan | MATH-500 | Minerva | OlympiadBench | Avg |
|---|---|---|---|---|---|---|---|---|---|
| base model, no fine-tuning | 0.079 | 0.388 | 0.079 | 0.287 | 0.217 | 0.653 | 0.301 | 0.275 | 0.285 |
| random_15k | 0.083 | 0.356 | 0.075 | 0.203 | 0.296 | 0.642 | 0.248 | 0.273 | 0.272 |
| cleaned_random | 0.054 | 0.394 | 0.092 | 0.202 | 0.288 | 0.665 | 0.250 | 0.293 | 0.280 |
| **cleaned_difficulty_weighted** (final) | 0.075 | 0.397 | 0.113 | 0.210 | 0.313 | 0.656 | 0.250 | 0.297 | **0.289** |
| cleaned_difficulty_topic_balanced | 0.088 | 0.384 | 0.067 | 0.190 | 0.305 | 0.653 | 0.243 | 0.308 | 0.280 |
| cleaned_hardest_only | 0.079 | 0.322 | 0.117 | 0.193 | 0.279 | 0.661 | 0.254 | 0.304 | 0.276 |

What we learned from these numbers:
- Just cleaning the data helps. cleaned_random averages 0.565, compared with 0.548 for random_15k. The gains are
  on CN_MATH (0.367 compared with 0.200), OlympiadBench (0.554 compared with 0.499) and Kaoyan (0.693 compared with
  0.668), while AIME goes down (0.200 compared with 0.300).
- Picking more of the problems the base model struggles with raises the average again. cleaned_difficulty_weighted
  averages 0.582, with its biggest gains over cleaned_random on AMC (0.775 compared with 0.675) and CN_MATH (0.433
  compared with 0.367).
- The four cleaned sets are close, though, and the average leans on the small benchmarks. AIME and CN_MATH have only
  30 questions each, so one question moves their score by 0.033 and the average by about 0.004. Counting all 1,944
  questions together instead, the four cleaned sets each solve between 1,247 and 1,258 of them at least once in 8
  tries, and cleaned_difficulty_weighted is the lowest of the four. All four are clearly ahead of random_15k (1,210)
  and the base model (1,181) either way. We trained each set only once and picked the final model by the per-benchmark
  average, which is how the leaderboard ranks, so the gaps between the four cleaned sets may be noise.
- cleaned_difficulty_topic_balanced (0.570) and cleaned_hardest_only (0.556) average a bit lower than
  cleaned_difficulty_weighted. For cleaned_hardest_only our guess is that it has too many problems that are too hard
  for a 3B model to learn from: up to 8,447 of its 15,000 rows are problems the base model got wrong in all 4 of its
  tries, and the rest are long problems we never measured.
- random_15k's higher AIME score (9 of 30, where most other models got 7) comes from two questions where only 1
  of its 8 answers was right, so it's mostly luck.
- pass@1 hardly changes (0.285 for the base model, 0.289 for cleaned_difficulty_weighted). Counting all 1,944
  questions together, it's 0.367 for the base model and between 0.368 and 0.371 for the four cleaned sets. So
  fine-tuning mainly gets more questions right at least once in 8 tries; a single answer isn't much more reliable.
- Every fine-tuned model does worse than the base model on GPQA (0.737 down to between 0.672 and 0.717) and Minerva
  (0.482 down to between 0.430 and 0.471). We didn't look into why, but two things may be related. Our training sets
  are all or mostly math. And the fine-tuned models write much longer answers than the base model (that's why their
  evals took so long, see Evaluation), so more of their answers may hit the eval's limit of 32,768 tokens before
  reaching a final answer.

[results/training_loss.png](results/training_loss.png) has the training loss of all five runs, smoothed with a moving
average. The drops at steps 1,875 and 3,750 are where epochs 2 and 3 start.

## What's in this repo

| folder | what's in it |
|---|---|
| `scripts/` | everything we wrote: data analysis, selection, training and eval. Each file starts with a comment that says what it does and how to run it. |
| `configs/` | the training config, and the exact package versions of our three Python envs |
| `subsets/` | the row numbers of each 15K training set, enough to rebuild them exactly |
| `results/` | scores of the base model and our five fine-tuned models, and the training loss plot |
Our scripts run inside the course starter repo
([reds-lab/Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM), commit 638989b), which
has the course eval (answers generated with vLLM and checked by a rule-based math grader) and LLaMA-Factory 0.9.4.dev0,
the fine-tuning library we train with. We didn't change either. Its README, the assignment handout, is what we mean
by "the course README".

## How we picked the data

### What's in the 100K rows

Scripts: `analyze_data.py` (lengths, answer format, give-up endings, overlap with the benchmarks), `link_and_grade.py`
(finds each math row in its source dataset and grades its final answer), `cluster_prompts.py` (sentence embeddings,
topic clusters, near duplicates) and `analysis_report.py` (joins those results and prints the tables).

- 67,432 math rows (from two public math datasets, OpenMathReasoning and NuminaMath-CoT) and 32,568 code rows.
- Every row already uses the eval's system prompt, "Please reason step by step, and put your final answer within
  \boxed{}." (`\boxed{}` is the LaTeX box the grader looks for.) Each solution is the reasoning, then a short written
  answer after a `</think>` tag.
- 3.7% of the math rows are longer than 16,384 tokens, the limit for one training example, so training cuts them off
  before the final answer.
- Some solutions end by giving up ("given that I'm stuck, I'll hazard a guess"). That gets more common the longer the
  solution is: 0.1% under 2K tokens, 41% between 14K and 16.4K.
- We matched the math rows to their source datasets, which gives most of them a reference answer, and graded each
  solution's final answer with the eval's own grader. Of the 25,126 rows where both answers are plain numbers, 93.5%
  of the solutions' answers are right. For other answer types the grader often marks equivalent answers as
  different, so we only trust a "wrong" verdict when both answers are plain numbers or option letters.
- 21.9% of the math rows share their exact question with another row, and 26.2% have a near duplicate (cosine
  similarity of 0.95 or more between all-MiniLM-L6-v2 sentence embeddings of the questions).
- 120 rows have a run of 50 characters in common with a benchmark question. Most of those are just common phrasing;
  only 20 contain at least half of a benchmark question. The cleaning below drops all of them anyway.

### The cleaned pool

Script: `select_subsets.py`. The cleaned pool is the set of math rows that pass all of the checks below, and every
training set except random_15k is drawn from it. We applied the checks in this order:

| check | math rows left |
|---|---:|
| all math rows | 67,432 |
| the whole example fits in 16,384 tokens | 64,907 |
| exactly one \boxed{} in the answer written after the reasoning (the eval only reads the last one) | 61,069 |
| doesn't end by giving up | 55,505 |
| doesn't say the problem is broken or that its answer is likely wrong | 54,598 |
| its final answer doesn't disagree with the reference answer (checked when both are plain numbers or option letters) | 53,653 |
| not in the 1% most repetitive solutions (measured by how much the text compresses) | 53,299 |
| no overlap with any benchmark question | 53,197 |
| one solution per question (the shortest verified correct one if there is one, else the shortest) | 47,430 |

### How hard each problem is for the base model

Script: `measure_difficulty.py`. The course README suggests measuring difficulty by generating several answers and
checking how many are right. We did that with the model we fine-tune. For the 20,317 pool problems whose final answer
is a plain number or option letter, the base Qwen2.5-3B-Instruct got 4 tries each, with the eval's prompt and
sampling (temperature 0.6, top-p 0.95) but answers capped at 4,096 tokens. A try counts as right when its final
answer matches the final answer of the dataset's own solution.

We split the problems into 6 shards (parts of the list) and ran each shard as its own job on an A30 GPU. Shard 3's
job ran out of time while grading its answers, and so did a second try that only redid the grading, so its 3,386
problems count as unmeasured. The other 16,931 came out like this:

| right tries | 0 of 4 | 1 of 4 | 2 of 4 | 3 of 4 | 4 of 4 |
|---|---:|---:|---:|---:|---:|
| problems | 8,447 | 2,584 | 1,798 | 1,545 | 2,557 |

A problem's solve rate is the share of its 4 tries that were right. The 30,499 pool problems without one (shard 3's,
plus the ones whose final answer isn't a plain number or letter) get an estimated solve rate instead: the average
solve rate of the measured problems with a similar length. Problems with longer solutions turned out to be harder:
the average solve rate goes from 0.69 for the shortest tenth to about 0.23 for the longer half.

### The five training sets

| name | how it's built |
|---|---|
| random_15k | 15,000 rows drawn uniformly at random from all 100,000 (seed 42). The assignment's baseline. |
| cleaned_random | 15,000 rows drawn uniformly at random from the cleaned pool. |
| **cleaned_difficulty_weighted** (final) | A weighted random draw (without replacement) from the cleaned pool. The weight depends on the base model's solve rate: 1.25 for 0 of 4, 1.5 for 1 or 2 of 4, 0.75 for 3 of 4 and 0.25 for 4 of 4. An estimated solve rate is grouped the same way: under 0.125, 0.125 to 0.625, 0.625 to 0.875, and 0.875 or more. So the problems it solved 1 or 2 times out of 4 get the most weight, the ones it never solved a bit less, and a problem it gets wrong at least half the time has 5 to 6 times the weight of one it always solves. |
| cleaned_difficulty_topic_balanced | Same weights, plus three things: all multiple choice problems go in first (GPQA is multiple choice), the rest is spread over 300 topic clusters (k-means on the question embeddings, each cluster's quota grows with the square root of its size), and a question is skipped if it's a near duplicate (cosine similarity 0.95 or more) of one already picked. |
| cleaned_hardest_only | The 15,000 problems with the lowest solve rate (longer solution first on ties), near duplicates skipped. This follows the idea of the LIMO paper ("LIMO: Less is More for Reasoning"), which fine-tunes on a small set of hard problems. Only 8,447 pool problems were measured at 0 of 4, so at least 6,553 of its rows are long problems we never measured, whose estimated solve rate is about 0.23. |

What ended up in each set:

| | random_15k | cleaned_random | cleaned_difficulty_weighted | cleaned_difficulty_topic_balanced | cleaned_hardest_only |
|---|---:|---:|---:|---:|---:|
| math rows | 68% | 100% | 100% | 100% | 100% |
| mean tokens per example (counting at most 16,384) | 7,482 | 6,442 | 6,646 | 6,466 | 6,583 |
| estimated base model solve rate | not measured | 0.30 | 0.26 | 0.26 | 0.10 |
| ends by giving up or gets cut off | 11.1% | 0% | 0% | 0% | 0% |
| multiple choice | 1.3% | 1.8% | 1.6% | 5.7% | 2.8% |
| repeated questions | 438 | 0 | 0 | 0 | 0 |

## Training

Config: [configs/sft_config.yaml](configs/sft_config.yaml). Script: `scripts/train.sh`. The training hyperparameters
are the ones in the course README's sample config. We only changed how the job runs: DeepSpeed stage, attention
kernel, Liger kernels, preprocessing workers, logging and saving, no wandb, and a local copy of the model.

| setting | value |
|---|---|
| base model | Qwen2.5-3B-Instruct, full fine-tuning (all weights) |
| framework | LLaMA-Factory 0.9.4.dev0, `qwen` chat template, the dataset's system prompt |
| epochs | 3 (5,625 optimizer steps) |
| batch | 8 sequences per optimizer step: 1 at a time on the GPU, with the gradients added up over 8 (gradient accumulation) |
| learning rate | 5e-6, cosine schedule, no warmup |
| optimizer | AdamW (beta1 0.9, beta2 0.999, eps 1e-8), no weight decay, gradient clipping at 1.0 (Transformers defaults, not set in our config) |
| max sequence length | 16,384 tokens (only random_15k has longer rows, and those get cut) |
| precision | bf16 |
| seed | 42 (the Transformers default) |
| memory | DeepSpeed ZeRO-2 on one GPU (LLaMA-Factory's `ds_z2_config.json`), Liger kernels (fused GPU code that saves memory, the loss included), PyTorch's built-in attention (SDPA) |
| hardware | one NVIDIA B200 per model (ARC's Owl cluster, from the job logs), 10.7 to 13.2 hours per model in two 8 hour jobs |

A run saves a checkpoint every 300 steps, and the next 8 hour job on the same cluster picks up from the last one.

## Evaluation

We used the course eval in `eval/` with its own settings: vLLM 0.6.1, temperature 0.6, top-p 0.95, 8 answers per
question, up to 32,768 new tokens per answer, the system prompt above, and rule-based grading of the last \boxed{}
answer.

- The base model ran with the course's `eval_single.sh`, one job per benchmark.
- The fine-tuned models' long answers caused two problems.
  1. vLLM 0.6.1 crashes when its KV cache (the GPU memory that holds the answers being written) fills up with 8 long
     answers per question. 27 of our first 40 runs died with `assert len(seqs) == 1` or "Aborted due to the lack of
     CPU swap space". `scripts/vllm_fix/sitecustomize.py` changes only how vLLM uses memory: it pauses requests by
     swapping them to CPU memory, and it caps how many run at once so that even 32K token answers fit. Python loads
     a file with this name at startup when its folder is on `PYTHONPATH` (`eval_worker.sh` sets that up), so eval.py
     itself doesn't change. Prompts, sampling and grading are untouched.
  2. One benchmark can take more than a day on one GPU (OlympiadBench is 675 questions x 8 long answers). So we cut
     each benchmark into slices of 2 to 3 GPU hours with eval.py's own `--start_idx` / `--end_idx`, ran the slices
     on whatever GPUs were free (A30, L40S, A100, H200), and combined the results. pass@8 and pass@1 are both
     averages over questions, so combining the slices gives the same numbers as one full run.
     `merge_eval_slices.py` grades every answer again with the course grader before writing the score.

AIME and CN_MATH for all five models, and AMC for three of them (random_15k, cleaned_difficulty_topic_balanced and
cleaned_hardest_only), ran as one job each with the course's `eval_single.sh`, before we had the slices. Everything
else ran as slices, 177 in total. Those one-job runs, and the base model's OlympiadBench run, had vLLM's CPU swap
space raised from 4 to 48 GiB through a one line change in eval.py. That's a memory setting only, and the course's
eval.py is otherwise untouched (the slices get the same setting from `vllm_fix/`).

There's one more difference from the course setup. Our eval env has antlr4-python3-runtime 4.9.3, while
`eval/requirements.txt` pins 4.11.1. With 4.11.1, latex2sympy2 (which the grader uses to read LaTeX) failed to
import for us. The note at the top of `configs/envs/evalenv.txt` has the details.

## Reproduce everything

You can skip the data analysis, because `subsets/` already has the row numbers of all five training sets. To retrain
our models, do steps 1, 2 and 3, build the five training files from those lists with the commands below (run from
the repo root), then do steps 8, 9 and 10:

```bash
cd scripts
for n in random_15k cleaned_random cleaned_difficulty_weighted cleaned_difficulty_topic_balanced cleaned_hardest_only; do
    ~/envs/sft/bin/python make_subset.py --name $n --rows ../subsets/${n}_rows.json
done
cd ..
```

Each step below says which folder it starts in.

### 0. What you need

- A Slurm cluster with GPUs. The scripts use ARC's names: the account `tml_2026`, the clusters TinkerCliffs (A100,
  H200), Owl (B200) and Falcon (A30, L40S), and partition and QoS names like `a100_normal_q` and
  `tc_a100_normal_short` (a partition is a group of nodes; a QoS, "quality of service", sets job limits and
  priority). To use another cluster, change them in the `#SBATCH` lines at the top of the scripts, in the `sbatch`
  commands of steps 5 and 6, in the `chain` lines at the end of `submit_training.sh`, in `LANES` and `LOGINS` in
  `eval_dispatcher.sh`, in the node name prefixes in `bad_nodes.sh`, and in the CUDA module names in `train.sh`.
- Conda (we used ARC's `Miniconda3` module), and a Hugging Face account (log in after step 1 with
  `~/envs/sft/bin/huggingface-cli login`).
- Disk: about 10 GB for the base model and the data, about 100 GB in `/scratch` per training run (two checkpoints of
  47 GB), and 7 GB per finished model in `$HOME/models`.

### 1. Code and environments (starts in any folder, ends in the repo root)

Our files go into a copy of the course starter repo. From here on, "the repo root" means that copy.

```bash
git clone https://github.com/reds-lab/Project-Reasoning-SFT-LLM.git
git clone https://github.com/yashmoar11/Trustworthy-Machine-Learning.git
cd Project-Reasoning-SFT-LLM
git checkout 638989b
cp -r ../Trustworthy-Machine-Learning/{scripts,configs,subsets,results} .
mkdir -p scripts/logs eval/logs
module load Miniconda3
# conda-forge only, so you don't have to accept Anaconda's terms for its default channel
for env in sft sft-b200 evalenv select; do conda create -y -p ~/envs/$env -c conda-forge --override-channels python=3.10; done

# training: one env with PyTorch built for CUDA 12.6 (A100, H200) and one for CUDA 12.8 (B200); train.sh picks the right one
~/envs/sft/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
~/envs/sft-b200/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
for env in sft sft-b200; do
    ~/envs/$env/bin/pip install -r configs/envs/sft.txt
    ~/envs/$env/bin/pip install --no-deps -e LLaMA-Factory
done
# eval (vLLM 0.6.1 and the grader)
~/envs/evalenv/bin/pip install -r configs/envs/evalenv.txt
~/envs/evalenv/bin/pip install --no-deps latex2sympy2==1.9.1
# selection (sentence embeddings and k-means, CPU only)
~/envs/select/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
~/envs/select/bin/pip install -r configs/envs/select.txt
```

The version files are the `pip freeze` output of the envs we actually used. The comment at the top of each one says
what's left out of it.

### 2. Base model and data (repo root)

```bash
~/envs/sft/bin/huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir eval/models/Qwen2.5-3B-Instruct
~/envs/sft/bin/huggingface-cli download redsgnaoh/acereason11_100k --repo-type dataset \
    --local-dir eval/train_data/acereason11_100k
```

Our scripts read the dataset's parquet files (`data/train-*.parquet`). They hold the same 100,000 rows as
`acereason11_100k.json` and load much faster. Row numbers everywhere in this repo are positions in those files.

### 3. Score the base model (repo root)

Submit from a shell with no conda env active (see "Problems we ran into").

```bash
cd eval
for b in aime amc cn_math_2024 gpqa kaoyan math minerva olympiadbench; do
    CONDA_ENV=$HOME/envs/evalenv MODEL=$PWD/models/Qwen2.5-3B-Instruct OUTPUT_DIR=./outputs/base_model \
        sbatch --job-name=base_$b eval_single.sh $b
done
cd ..
```

If OlympiadBench dies with "Aborted due to the lack of CPU swap space" (ours did), resubmit it from `eval/` with our
vLLM memory settings. Besides the bigger swap space, they also change how vLLM pauses requests, but not the prompts,
sampling or grading:

```bash
cd eval
PYTHONPATH=$PWD/../scripts/vllm_fix EVAL_VLLM_FIX=1 VLLM_SWAP_SPACE=48 \
CONDA_ENV=$HOME/envs/evalenv MODEL=$PWD/models/Qwen2.5-3B-Instruct OUTPUT_DIR=./outputs/base_model \
    sbatch --job-name=base_olympiadbench eval_single.sh olympiadbench
cd ..
```

### 4. Random baseline (repo root, then `scripts/` until step 9)

```bash
cd scripts
~/envs/sft/bin/python make_subset.py --name random_15k --random 15000
```

This writes the training file `LLaMA-Factory/data/random_15k.json` and the row list
`subsets/random_15k_rows.json`, and registers the dataset in `LLaMA-Factory/data/dataset_info.json`.

### 5. Analyze the data (`scripts/`, CPU jobs)

The first three jobs don't depend on each other, so they can run at the same time. `analysis_report.py` needs all
three, plus `subsets/random_15k_rows.json` from step 4, and it writes `eval/train_data/merged.parquet`, which
`select_subsets.py` reads.

```bash
sbatch -A tml_2026 -p normal_q --cpus-per-task=16 --mem=64G --time=1:00:00 --output=logs/analyze_%j.out \
    --wrap="$HOME/envs/sft/bin/python analyze_data.py"
sbatch -A tml_2026 -p normal_q --cpus-per-task=32 --mem=64G --time=2:00:00 --output=logs/link_%j.out \
    --wrap="$HOME/envs/evalenv/bin/python link_and_grade.py"
sbatch -A tml_2026 -p normal_q --cpus-per-task=32 --mem=64G --time=1:00:00 --output=logs/cluster_%j.out \
    --wrap="$HOME/envs/select/bin/python cluster_prompts.py"
# once those are done:
~/envs/select/bin/python analysis_report.py > logs/analysis_report.txt
~/envs/select/bin/python select_subsets.py --candidates
```

`analysis_report.py` prints the tables behind the numbers above. `select_subsets.py --candidates` writes the list of
problems that step 6 measures (the pool problems whose final answer is a plain number or option letter).

### 6. Measure difficulty (`scripts/`, 6 GPU jobs, about an hour each on an A30)

```bash
for i in 0 1 2 3 4 5; do
    sbatch -M falcon -A tml_2026 -p a30_normal_q --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 \
        --output=logs/difficulty_%j.out --wrap="$HOME/envs/evalenv/bin/python measure_difficulty.py --shard $i --num_shards 6"
done
```

`select_subsets.py` leaves out shard 3 on purpose (the `DIFFICULTY_SHARDS` list near the top of the script), because
that's what our training sets were built with. Add 3 back if you'd rather use all of them. If a shard's job runs out
of time while grading, `measure_difficulty.py --shard <i> --num_shards 6 --regrade` grades its saved answers again on
a CPU.

### 7. Build the four training sets (`scripts/`)

```bash
~/envs/select/bin/python select_subsets.py
for n in cleaned_random cleaned_difficulty_weighted cleaned_difficulty_topic_balanced cleaned_hardest_only; do
    ~/envs/sft/bin/python make_subset.py --name $n --rows ../subsets/${n}_rows.json
done
git status ../subsets   # no changes means you got exactly our rows
```

If you redo steps 5 and 6 from scratch, the four selected sets can come out a little different: the source datasets
on the Hugging Face Hub may have changed since (we didn't pin a version), and sampling and the grader's 5 second time
limit can behave differently on other machines. The lists in `subsets/` are the ones we trained on.

### 8. Train (`scripts/`, on a login node)

```bash
for n in random_15k cleaned_random cleaned_difficulty_weighted cleaned_difficulty_topic_balanced cleaned_hardest_only; do bash submit_training.sh $n; done
```

This queues each run on several GPU types at once, each as a chain of 8 hour jobs. Only one copy of a run trains at a
time (`job_lock.sh` makes sure of that). Later jobs on the same cluster resume from its last checkpoint, since the
checkpoints are on that cluster's `/scratch`, and every copy exits right away once the model is done. A run took two
8 hour jobs on a B200; slower GPUs need more (`submit_training.sh` chains 4 on A100). On a single GPU type you can
also submit `train.sh` yourself, chaining the jobs so each one resumes:

```bash
id=$(RUN=cleaned_difficulty_weighted sbatch --parsable -M <cluster> -p <gpu partition> --qos=<qos> train.sh)
RUN=cleaned_difficulty_weighted sbatch -M <cluster> -p <gpu partition> --qos=<qos> --dependency=afterany:${id%%;*} train.sh
```

Checkpoints go to `/scratch/$USER/saves/<name>`, and the finished model is copied to `$HOME/models/<name>`.

### 9. Evaluate the fine-tuned models (`eval/`)

Wait until all five models are in `$HOME/models` (each folder has a `config.json`). Then:

```bash
cd ../eval
~/envs/evalenv/bin/python ../scripts/make_eval_slices.py      # writes scripts/eval_slices.txt (190 slices)
sbatch -M tinkercliffs ../scripts/eval_dispatcher.sh
```

That's 38 slices per model. AIME, CN_MATH and AMC are one slice each, so the 190 cover both our 177 slices and the 13
runs we did as one job. The dispatcher is a small CPU job. It keeps eval workers (`eval_worker.sh`) queued on every
GPU type until all slices are done, and every 15 minutes it merges finished benchmarks into
`outputs/<name>/log_<benchmark>.txt`. Its log, `eval/logs/eval_dispatcher_<job id>.out`, shows progress. Without the
dispatcher you can submit workers by hand,
`sbatch -M <cluster> -p <gpu partition> --qos=<qos> --time=8:00:00 ../scripts/eval_worker.sh`, then run
`~/envs/evalenv/bin/python ../scripts/merge_eval_slices.py` at the end. Each slice gets up to 3 tries.

### 10. Score table and loss plot (`eval/`)

```bash
~/envs/evalenv/bin/python ../scripts/collect_results.py outputs --csv ../results/scores.csv
cd ../scripts && ~/envs/sft/bin/python plot_training_loss.py
```

## Problems we ran into, and how we got around them

- If a conda env is active when you submit a Slurm job, `source activate` inside the job falls back to the base
  Python without any error. Our scripts call each env's Python by its path and clear the inherited conda variables.
- LLaMA-Factory needs the `system` column mapped in `dataset_info.json`, or it trains with Qwen's default system
  prompt instead of the eval's. `make_subset.py` maps it.
- LLaMA-Factory saves `tokenizer.json` in a newer format than vLLM 0.6.1 can read. `fix_tokenizer.sh` copies the base
  model's tokenizer files back (training doesn't change the tokenizer), and `train.sh` and `eval_worker.sh` run it.
- On ARC, a compute node can only submit jobs to its own cluster, so the dispatcher submits Falcon jobs through a
  TinkerCliffs login node over ssh.
- `squeue` keeps listing a job for about 5 minutes after it ends, so when our scripts check whether a job is still
  running, they ask `squeue` for pending and running jobs only (`-t PD,R`).
- When a job hits its time limit, Slurm kills it before bash can run the job's cleanup code (its exit trap), so the
  lock that `job_lock.sh` made for it stays behind. `job_lock.sh` treats a lock whose owner is gone as free.
- A few GPUs were stuck in hardware slowdown (the GPU cuts its own clock speed) and ran several times slower than
  normal. `train.sh` and `eval_worker.sh` check for it at startup, quit, and write the node to `~/.bad_gpu_nodes`.
  `bad_nodes.sh` turns that file into an `--exclude` option, so later submissions skip those nodes.
- Two of our 177 slices ran twice at the same time, because of a bug in `job_lock.sh` that let two workers take the
  same slice at the same moment (fixed in this repo). Both copies finished, and eval.py replaces its output file in
  one step, so the merge used the one written last.

## Notes on the eval set

- All 198 GPQA questions in the course's file have "A" as the right answer, because the options aren't shuffled. We
  didn't use that anywhere. Each question has 4 options, so guessing a letter at random 8 times would get at least
  one right on about 90% of them, more than any model here scores. So GPQA's pass@8 tells us little.
