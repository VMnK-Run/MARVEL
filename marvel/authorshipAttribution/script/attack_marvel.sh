attack_type="alert"
save_name="marvel"
model_type="codebert"

slip="_"
log_name=$attack_type$slip$model_type$slip$save_name

CUDA_VISIBLE_DEVICES=0 python ../code/attack.py \
    --attack_name=$attack_type \
    --model_type=$model_type \
    --eval_batch_size=8 \
    --save_name=$save_name 2>&1 | tee $log_name.log