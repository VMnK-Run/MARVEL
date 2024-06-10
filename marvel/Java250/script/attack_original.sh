attack_type="alert"
save_name="original"
model_type="codebert"
left=0
right=500

slip="_"
log_name=$attack_type$slip$model_type$slip$save_name$slip$left$slip$right

CUDA_VISIBLE_DEVICES=1 python ../code/attack.py \
    --attack_name=$attack_type \
    --model_type=$model_type \
    --eval_batch_size=8 \
    --index $left $right \
    --save_name=$save_name 2>&1 | $log_name.log