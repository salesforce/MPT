export TASK_NAME=glue
export DATASET_NAME=paws-x

bs=16  # xlm-r-large: 32
lr=$3
dropout=0.1
psl=$5
epoch=40   ## orginal: 40
model=$1
GPU=$2
#seed=11
seed=$4

CUDA_VISIBLE_DEVICES=${GPU} python3 run.py \
  --model_name_or_path $1 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --lang en \
  --max_seq_length 128 \
  --save_steps 20000 \
  --log_level error \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-${model}-${psl}-${lr}-${seed}_40/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed ${seed} \
  --save_total_limit  1 \
  --log_level error \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --prefix
