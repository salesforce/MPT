export TASK_NAME=pos
export DATASET_NAME=udpos


bs=16
epoch=30
psl=$4
lr=$3
dropout=0.1
model=$1
GPU=$2
seed=$5 # 11

CUDA_VISIBLE_DEVICES=${GPU} python3 run.py \
  --model_name_or_path $model \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --lang en \
  --max_seq_length 152 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-${model}-${psl}-${lr}-${seed}/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed ${seed} \
  --save_total_limit  1 \
  --log_level error \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --prefix
