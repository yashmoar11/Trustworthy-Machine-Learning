# vLLM 0.6.1 memory settings for the course eval. At startup Python imports any file named sitecustomize.py on its
# path, and eval_worker.sh puts this folder on PYTHONPATH. The patch only runs when EVAL_VLLM_FIX=1, which
# eval_worker.sh also sets. eval.py, its prompts, sampling and grading stay the same; this only changes how vLLM fits
# the running requests into memory:
#   - The fine-tuned models write long answers (up to the eval's limit of 32,768 tokens), so the KV cache (the GPU
#     memory that holds the answers being written) fills up and vLLM pauses requests (vLLM calls this preemption).
#     A request whose other samples (n=8) already finished gets paused by "recompute" (its cache is thrown away and
#     rebuilt later), which vLLM 0.6.1 can't resume ("assert len(seqs) == 1" in Scheduler.schedule). So we pause by
#     swapping to CPU memory instead.
#   - Paused requests can then outgrow the CPU swap space ("Aborted due to the lack of CPU swap space"). So we cap
#     the number of sequences running at once at what fits in GPU + CPU memory even if every one hits 32K tokens.
#   - CPU swap space is $VLLM_SWAP_SPACE GiB. vLLM's default of 4 GiB is too small for answers this long, so
#     eval_worker.sh sets 48.
import os


def _patch():
    import vllm
    from vllm import LLM

    if not vllm.__version__.startswith("0.6.1"):
        return
    original_init = LLM.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("preemption_mode", "swap")
        kwargs.setdefault("swap_space", int(os.environ.get("VLLM_SWAP_SPACE", 4)))
        original_init(self, *args, **kwargs)
        engine = self.llm_engine
        cache = engine.cache_config
        blocks_per_seq = -(-engine.model_config.max_model_len // cache.block_size) + 1
        fit = int(0.95 * (cache.num_gpu_blocks + cache.num_cpu_blocks)) // blocks_per_seq // 8 * 8
        engine.scheduler_config.max_num_seqs = max(8, min(engine.scheduler_config.max_num_seqs, fit))
        print(f"[vLLM 0.6.1 settings] preemption by swap; {cache.num_gpu_blocks} GPU + {cache.num_cpu_blocks} CPU blocks "
              f"of {cache.block_size} tokens; at most {engine.scheduler_config.max_num_seqs} sequences at once", flush=True)

    LLM.__init__ = __init__


if os.environ.get("EVAL_VLLM_FIX") == "1":
    _patch()
