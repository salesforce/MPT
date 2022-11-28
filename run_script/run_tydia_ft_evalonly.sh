export TASK_NAME=qa
export DATASET_NAME=tydia
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=5e-3
dropout=0
psl=16
epoch=30
checkpoint=$1

echo
echo "TydiQA-GolP"
for lang in en ar bn fi id ko ru sw te; do
#for lang in en; do
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
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy epoch \
  --evaluation_strategy epoch
  echo -n "  $lang "
  echo
done


