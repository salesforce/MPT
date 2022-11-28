export TASK_NAME=pos
export DATASET_NAME=udpos
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=5e-3
dropout=0
epoch=30
checkpoint=$1

# 'af', 'ar', 'az', 'bg', 'bn', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'lt', 'ml', 'mr', 'ms', 'my', 'nl', 'pa', 'pl', 'pt', 'qu', 'ro', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yo', 'zh'

## pos   af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh
echo
echo "UD pos finetune"
# ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu qu pl uk az lt pa gu ro
for lang in af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh; do
  echo
  echo -n "  $lang "
  echo
  python3 run.py \
  --model_name_or_path xlm-roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --resume_from_checkpoint  $checkpoint  \
  --do_eval \
  --do_predict \
  --lang $lang \
  --log_level error \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir checkpoints/$DATASET_NAME/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
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
