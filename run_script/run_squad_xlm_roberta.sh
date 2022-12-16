export TASK_NAME=qa
export DATASET_NAME=squad

bs=8  #xlm-r-large: 8
#lr=5e-3
dropout=0.2
psl=$4
epoch=30
model=$1
lr=$2
GPU=$3
seed=$5 ##11



CUDA_VISIBLE_DEVICES=${GPU} python3 run.py \
  --model_name_or_path $1 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/${DATASET_NAME}_${model}_${lr}_${psl}_${seed}/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed ${seed} \
  --save_total_limit  1 \
  --log_level error \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --prefix
