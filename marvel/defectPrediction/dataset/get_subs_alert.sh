python get_substitutes_alert.py \
    --store_path ./substitutions/graphcodebert_test_subs_0_1000.jsonl \
    --base_model=../../../pretrained_model/graphcodebert-base \
    --eval_data_file=./valid.txt \
    --index 0 1000 --block_size 512