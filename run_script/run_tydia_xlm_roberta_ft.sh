export TASK_NAME=qa
export DATASET_NAME=tydia

bs=8 #8
#lr=5e-3
dropout=0
psl=16
epoch=30
model=$1
lr=$2
GPU=$3
seed=$4 # 11


# --gradient_accumulation_steps 2 \

CUDA_VISIBLE_DEVICES=${GPU} python3 run.py \
  --model_name_or_path $1 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --lang en \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir checkpoints/${DATASET_NAME}_${model}_${lr}_finetune_${seed}/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed ${seed} \
  --save_total_limit  1 \
  --log_level error \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end
