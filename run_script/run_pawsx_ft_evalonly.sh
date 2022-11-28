export TASK_NAME=glue
export DATASET_NAME=paws-x
export CUDA_VISIBLE_DEVICES=0

bs=32
lr=5e-3
dropout=0.2
psl=16  # 32
epoch=30
checkpoint=$1

echo
echo "PAWS-X"
for lang in en de es fr ja ko zh; do
  echo
  echo -n "  $lang "
  echo
  python3 run.py \
  --model_name_or_path xlm-roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --resume_from_checkpoint  $checkpoint  \
  --do_eval \
  --lang $lang \
  --log_level error \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir checkpoints/$DATASET_NAME/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --max_seq_length 128  \
  --save_strategy epoch \
  --evaluation_strategy epoch
  echo -n "  $lang "
  echo
done


#python3 run.py \
#  --model_name_or_path roberta-large \
#  --task_name $TASK_NAME \
#  --dataset_name $DATASET_NAME \
#  --resume_from_checkpoint  checkpoints/squad/checkpoint-298917  \
#  --do_eval \
#  --lang en \
#  --per_device_train_batch_size $bs \
#  --learning_rate $lr \
#  --num_train_epochs $epoch \
#  --pre_seq_len $psl \
#  --output_dir checkpoints/$DATASET_NAME/ \
#  --overwrite_output_dir \
#  --hidden_dropout_prob $dropout \
#  --seed 11 \
#  --save_strategy epoch \
#  --evaluation_strategy epoch \
#  --prefix
