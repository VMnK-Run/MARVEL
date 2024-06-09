save_name="marvel"
model_type="codebert"

slip="_"
log_name=$model_type$slip$save_name

CUDA_VISIBLE_DEVICES=0 python ../code/run_mutual.py \
        --do_train \
        --do_test \
        --epochs=30 \
        --model_type $model_type \
        --save_name $save_name \
        --train_batch_size=8 \
        --eval_batch_size=8 \
        --alpha=0.3 \
        --max_adv_step=3 2>&1 | tee ../log/train_log/$model_type/train_$log_name.log