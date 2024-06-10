python get_substitutes_alert.py \
    --store_path ./substitutions/codebert_valid_subs.jsonl \
    --base_model="microsoft/codebert-base-mlm" \
    --eval_data_file=./valid.txt \
    --block_size 512 