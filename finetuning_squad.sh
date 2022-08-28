export BERT_BASE_DIR=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/liubiao/PTMs/glue_data
export SQUAD_DIR=/data/liubiao/PTMs/bert/SQuAD1.1
python run_squad.py --vocab_file=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12/bert_model.ckpt --do_train=True --train_file=/data/liubiao/PTMs/bert/SQuAD1.1/train-v1.1.json --do_predict=True --predict_file=/data/liubiao/PTMs/bert/SQuAD1.1/dev-v1.1.json --train_batch_size=12 --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=384 --doc_stride=128 --output_dir=/data/liubiao/PTMs/bert/SQuAD1.1/squad_base_output