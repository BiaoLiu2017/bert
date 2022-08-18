export BERT_BASE_DIR=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/liubiao/PTMs/glue_data
export FINETUNING_DIR=/data/liubiao/PTMs/bert/finetuning_dir
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$FINETUNING_DIR/mrpc_output/


#python run_classifier.py   --task_name=MRPC   --do_train=true   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=$FINETUNING_DIR/mrpc_output/

#bert_config.json
#{
#  "attention_probs_dropout_prob": 0.1,
#  "hidden_act": "gelu",
#  "hidden_dropout_prob": 0.1,
#  "hidden_size": 768,
#  "initializer_range": 0.02,
#  "intermediate_size": 3072,
#  "max_position_embeddings": 512,
#  "num_attention_heads": 12,
#  "num_hidden_layers": 12,
#  "type_vocab_size": 2,
#  "vocab_size": 30522
#}

#INFO:tensorflow:evaluation_loop marked as finished
#I0809 10:54:28.544660 139639954888448 error_handling.py:101] evaluation_loop marked as finished
#INFO:tensorflow:***** Eval results *****
#I0809 10:54:28.544880 139639954888448 run_classifier.py:940] ***** Eval results *****
#INFO:tensorflow:  eval_accuracy = 0.8455882
#I0809 10:54:28.544974 139639954888448 run_classifier.py:942]   eval_accuracy = 0.8455882
#INFO:tensorflow:  eval_loss = 0.50463086
#I0809 10:54:28.545222 139639954888448 run_classifier.py:942]   eval_loss = 0.50463086
#INFO:tensorflow:  global_step = 343
#I0809 10:54:28.545322 139639954888448 run_classifier.py:942]   global_step = 343
#INFO:tensorflow:  loss = 0.50463086
#I0809 10:54:28.545399 139639954888448 run_classifier.py:942]   loss = 0.50463086