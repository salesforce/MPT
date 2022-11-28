export TASK_NAME=qa
export DATASET_NAME=squad

bs=4 ## xlm-r-large: 8
#lr=5e-3
dropout=0
psl=4
epoch=2 #2 
model=$1
lr=$2
GPU=$3
seed=$4



CUDA_VISIBLE_DEVICES=${GPU} python3 run.py \
  --model_name_or_path $1 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/${DATASET_NAME}_${model}_${lr}_${seed}_finetune2/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed ${seed} \
  --save_total_limit  1 \
  --log_level error \
  --save_strategy epoch \
  --evaluation_strategy epoch
