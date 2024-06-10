save_name="original"
model_type="codebert"

slip="_"
log_name=$model_type$slip$save_name

python ../code/run_original.py \
    --do_train \
    --do_test \
    --model_type=$model_type \
    --epoch 5 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --save_name $save_name \
    --seed 123456 2>&1 | tee train_$log_name.log