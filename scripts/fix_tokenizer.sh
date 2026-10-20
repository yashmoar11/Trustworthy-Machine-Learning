#!/bin/bash
# Give a fine-tuned model the base model's tokenizer files and generation_config.json, and delete added_tokens.json,
# special_tokens_map.json and chat_template.jinja if they're there.
# LLaMA-Factory (tokenizers 0.21) saves tokenizer.json in a newer format than the eval env (vLLM 0.6.1 with
# tokenizers 0.19) can read. Fine-tuning doesn't change the tokenizer, so we just copy the original files back.
# Safe to run again, and from several jobs at once: each file is copied under a temporary name and then renamed with
# mv, so no job ever reads a half-copied file.
#   bash fix_tokenizer.sh $HOME/models/cleaned_difficulty_weighted
MODEL=${1:?usage: fix_tokenizer.sh <model dir>}
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../eval/models/Qwen2.5-3B-Instruct" && pwd)
[ "$(cd "$MODEL" && pwd)" = "$BASE" ] && exit 0
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do
    cmp -s "$BASE/$f" "$MODEL/$f" && continue
    cp "$BASE/$f" "$MODEL/.$f.tmp.$$" && mv -f "$MODEL/.$f.tmp.$$" "$MODEL/$f"
done
rm -f "$MODEL/added_tokens.json" "$MODEL/special_tokens_map.json" "$MODEL/chat_template.jinja"
