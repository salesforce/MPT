export TASK_NAME=glue
export DATASET_NAME=mnli

bs=16 # before: 32; xlm-rober-xl
lr=$3
dropout=0.1
psl=$4
epoch=30
model=$1
GPU=$2
seed=$5

CUDA_VISIBLE_DEVICES=${GPU} python3 -u run.py \
  --model_name_or_path $1 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train  \
  --do_eval \
  --max_seq_length 128 \
  --save_steps 20000 \
  --log_level error \
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


## --gradient_accumulation_steps 2
## --deepspeed deepspeed.json   --fp16
## --warmup_steps 36816
