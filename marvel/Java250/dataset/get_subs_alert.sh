python get_substitutes_alert.py \
    --store_path ./substitutions/graphcodebert_test_subs_0_500.jsonl \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=./test.txt \
    --index 0 500 --block_size 512 