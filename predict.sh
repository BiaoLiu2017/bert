export BERT_BASE_DIR=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/liubiao/PTMs/glue_data
export TRAINED_CLASSIFIER=/data/liubiao/PTMs/bert/finetuning_dir/mrpc_output/model.ckpt-343

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/data/liubiao/PTMs/bert/finetuning_dir/mrpc_output/