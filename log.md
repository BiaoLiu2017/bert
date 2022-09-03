# Flows
## 1. download & env & model  
git clone git@github.com:google-research/bert.git  
pip install tensorflow-gpu==1.15.0  
download models(links in bert github page)  
注：numpy最好是1.19.5版本，版本过高可能导致报错；  

## 2. download fine-tuning data  
bert-large need too large GPU RAM to reproduce paper。
so fine-tuning bert-base.
download glue data(execute the blow code)(有的网络可能受限导致无法下载，在本地windows下下载glue data的):
https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3

## 3. fine-tuning  
按照readme中的fine-tuning代码就能正常的fine-tuning了（注意修改bert路径以及glue_data路径）。

## 4. bert文本预处理  
- 类：tokenization.FullTokenizer；  
主要由basic_tokenizer和wordpiece_tokenizer两个类组成；  
- basic_tokenizer用于分词，步骤为：  
1）转换为unicode（即python中的str）；  
2）clean text（去特殊文本，包括不可见字符'\x00'，除\t\r\n外的控制字符；将空白符转化为空格' '，
即包括\n,\t,\r以及unicodedata.category为"Zs"的字符，转化为空格）  
3）处理中文符号（由于在英文wiki数据中也有少量的中文字符存在，因此在英文bert模型中也加入了中文字符处理函数，
具体操作是在CJK符号前后加上空格，CJK即中日韩越统一表意文字（CJKV Unified Ideographs））；  
4）分词（先取出两边的空白符，然后依据空白符来分词）  
5）（可选，如果do_lower_case为True则执行）小写化和去accents;(unicodedata.normalize("NFD", text)
即把'á'拆分成了'a'和́́''两个部分;对于unicodedata.category为'Mn'的去除，即去除了accents，结果就是将'á'变成了'a')  
6)去标点符号（此时可能存在如“a?”,"d,a"等字符，需要将标点符号和一般字符分开）；
效果：'abc$*de#f'分隔为了['abc', '$', '*', 'de', '#', 'f']  
标点符号定义为：(cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) 
or (cp >= 123 and cp <= 126))或cat = unicodedata.category(char)，cat.startswith("P"):  
7）最后再空格分词；whitespace_tokenize(" ".join(split_tokens))，即先用空格串连，再空格分词；  
- wordpieceTokenizer用于对basic_tokenizer分词之后的每个token进行再分词；  
原理：WPT本质上是从左到右，最长匹配原则（greedy longest-match-first algorithm），以bryant为例
先看bryant是否在vocab中，不在，看bryan是否在vocab中，直到发现br在vocab中；
剩余词为yant，为了告知model剩余的yant不是一个token的开头，因此加上##，即剩余词为##yant；
同样先看##yant是否在vocab，以此类推，最终bryant通过WPT分词，最终变为了[br, ##yan, ##t]

## 5. 生成InputExample
主要函数convert_single_example，包括以下步骤：
对两个文本利用tokenizer进行预处理和分词得到tokens_a和tokens_b，若长度超过max_seq_length-3，则每次去掉较长的最后一个token，直到满足条件；
然后得到
"input_ids"
"input_mask"
"segment_ids"
"label_ids"
"is_real_example"
并初始化InputFeatures对象；
```text
即：
# (a) For sequence pairs:
#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#  input_ids: 101 2105 6021 481 13938  2102 1010 102  66 20 39 25 163 102  0 0 0 0 0      #最后的0为[PAD]
#  input_mask: 1   1    1    1    1     1     1   1    1  1  1  1  1    1  0 0 0 0 0      #即1表示文本所在位置，0表示PAD  
#  segment_ids: 0  0    0    0    0     0     0   0    1  1  1  1  1    1  0 0 0 0 0      #即第一句话、第二句话、PAD
#  label_ids： 若label_map[label_ids]为1表示两句话为支持/蕴含关系，为0则表示独立关系。
# (b) For single sequences:
#  tokens:   [CLS] is this jack ##son ##ville ? [SEP]
#  input_ids: 101 2105 6021 481 13938  2102 1010 102 0 0 0 0 0      #最后的0为[PAD]
#  input_mask: 1   1    1    1    1     1     1   1  0 0 0 0 0      #即1表示文本所在位置，0表示PAD  
#  segment_ids: 0  0    0    0    0     0     0   0  0 0 0 0 0      #由于只有一句话，故全为0
#  label_ids： label_map[label_ids]为其类别标签；
```

## 6.写入tfrecord
Convert a set of `InputExample`s to a TFRecord file.即将样本写入tfrecord文件。  
实现函数为：
file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
具体写入的特征包括：
"input_ids"
"input_mask"
"segment_ids"
"label_ids"
"is_real_example"
```text
writer = tf.python_io.TFRecordWriter(output_file)
feature = convert_single_example(ex_index, example, label_list,
                                 max_seq_length, tokenizer)
def create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f
features = collections.OrderedDict()
features["input_ids"] = create_int_feature(feature.input_ids)
features["input_mask"] = create_int_feature(feature.input_mask)
features["segment_ids"] = create_int_feature(feature.segment_ids)
features["label_ids"] = create_int_feature([feature.label_id])
features["is_real_example"] = create_int_feature(
    [int(feature.is_real_example)])
tf_example = tf.train.Example(features=tf.train.Features(feature=features))
writer.write(tf_example.SerializeToString())
```
## 7. Train  
- 1)构建TPUEstimator
```text
estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
```
其中model_fn为构建模型的函数
```text
model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)
```
该函数能够提取数据：
```text
input_ids = features["input_ids"]
input_mask = features["input_mask"]
segment_ids = features["segment_ids"]
label_ids = features["label_ids"]
is_real_example = None
if "is_real_example" in features:
  is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
else:
  is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
```
并计算loss以及创建优化器：
```text
(total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
```
- 2)生成input_fn函数用于数据读取
input_fn函数包括的功能有读取tfrecord文件并还原为example字典，其key为"input_ids"，"input_mask"，"segment_ids"，"label_ids"，"is_real_example"。
并且针对train的情况会进行shuffle。以batch的形式返回数据。默认丢弃不够组成batch剩余的样本。

- 3)train
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


# 模型细节
## 1. 整体架构
函数为：
```text
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(...)
  ...
```
model为BertModel类的实例化对象，对于分类任务而言，需要[CLS]对应的vector（[batch_size, 768]），然后进过pooler，得到:  
output_layer = model.pooled_output  #[batch_size, 768]  
然后经过dropout：  
output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)#[batch_size, 768]  
然后再过一个全连接层：  
logits = tf.matmul(output_layer, output_weights, transpose_b=True)#如二分类，有 两个神经元[batch_size, 2]  
logits = tf.nn.bias_add(logits, output_bias)#[batch_size, 2]  
再计算p和log(p)：  
probabilities = tf.nn.softmax(logits, axis=-1)#[batch_size, 2]，即[p1, p2, ...]  
log_probs = tf.nn.log_softmax(logits, axis=-1)#[batch_size, 2]，即[logp1, logp2, ...]    
```text
即tf.nn.log_softmax的操作为：logits - log(reduce_sum(exp(logits), axis))
[a1,a2] - log(sum(exp([a1,a2])))
[a1,a2] - log(e^a1+e^a2)
[log[(e^a1)/(e^a1+e^a2)], log[(e^a2)/(e^a1+e^a2)]]
[log(p1), log(p2)]
```
对label进行one-hot编码：  
one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)#[batch_size, 2]  
求得每个样本的loss：
per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)#[batch_size, 1]
平均batch的loss：    
loss = tf.reduce_mean(per_example_loss)#[1,]    
因此最终返回的结果为：  
loss, per_example_loss, logits, probabilities #[1,] [batch_size, 1]  [batch_size, 2]  [batch_size, 2]  
- BertForSequenceClassification
对于序列分类任务，对应pytorch_transformers的BertForSequenceClassification类，源码为：  
```text
outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
pooled_output = outputs[1]#即[CLS]对应最后一层的hidden vec，经过一个全连接层，激活函数为Tanh。
pooled_output = self.dropout(pooled_output)
logits = self.classifier(pooled_output)#线性变换
```
```text
pooled_output来源：
first_token_tensor = hidden_states[:, 0]
pooled_output = self.dense(first_token_tensor)
pooled_output = self.activation(pooled_output)
```

## 2.Bert模型
```text
model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
```
bert的整体代码流程：  
1）embeddings
2）embedding_postprocessor
3）encoder（包括transformer_model和[CLS]对应vec的pooling）

### 2.1 embeddings
通过embedding_lookup函数进行embeddings。
```text
(self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)
```
构建一个embedding_table，用于查表索引得到token对应的embeddings，该表的参数初始化依然是采取truncated normal distribution。  
```text
embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))
```
将input_ids展平：  
flat_input_ids = tf.reshape(input_ids, [-1])#[batch_size*seq_length,]  
然后查表得到每个input_id对应的embeddings：  
对于TPU，先one_hot再matmul，比tf.gather更快，但效果和tf.gather是等价的：  
one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)  
output = tf.matmul(one_hot_input_ids, embedding_table)#得到input_ids对应的embedding，shape为(batch_size*seq_length,embedding_size)  
对于GPU/CPU，直接采取tf.gather更快：  
output = tf.gather(embedding_table, flat_input_ids)#得到input_ids对应的embedding，shape为(batch_size*seq_length,embedding_size)  
最终成功得到embeddings之后的向量：  
batch_size*seq_length,embedding_size

### 2.2 embedding_postprocessor 
即embeddings的后处理，函数为embedding_postprocessor：  
```text
self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,#[batch_size, seq_length, 1*embedding_size]
            use_token_type=True,
            token_type_ids=token_type_ids,#[batch_size, seq_length]，即该位置所属第一句还是第二句还是PAD
            token_type_vocab_size=config.type_vocab_size,#2，由于bert最多输入成对句子，因此，size是2
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,#0.02
            max_position_embeddings=config.max_position_embeddings,#512
            dropout_prob=config.hidden_dropout_prob)#0.1
``` 
1）segment embeddings  
首先加入segment embeddings，跟token embeddings一样，也是可学习的参数，构建segment embeddings的table，即token_type_table，  
用于查表得到对应的segment embeddings：  
```text
token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],#[2, 1*embedding_size]，embedding_size保持和token embedding一致
        initializer=create_initializer(initializer_range))
```
然后查表得到对应的embeddings，此时的vocab很小，一般为2，因此one-hot+matmul比tf.gather要快（无论TPU/GPU/CPU）
token_type_ids为输入的segment id，对应的shape为[batch_size, seq_length]，其值为0或者1.  
先展平，然后one-hot编码，再和table点积并reshape得到embeddings:  
flat_token_type_ids = tf.reshape(token_type_ids, [-1])#(batch_size*seq_length,)  
one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)#[batch_size*seq_length, 2]  
token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)#[batch_size*seq_length, 1*embedding_size]  
token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])#[batch_size, seq_length, 1*embedding_size]  
然后将token embeddings和segment embeddings合并：  
output += token_type_embeddings#合并token embedding和segment embedding，[batch_size, seq_length, 1*embedding_size]

2)position embeddings  
对于bert base，max_position_embeddings为512，即对512个位置进行embeddings，然后学习到每个位置的embeddins。  
跟前面一样，创建position对应的embeddins table:  
```text
full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],#[max_position_embeddings, embedding_size]
          initializer=create_initializer(initializer_range))      
```
由于实际上的seq_length一般比max_position_embeddings小，因此先进行slice，得到更细的表：    
position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])#[seq_length, 1*embedding_size]  
reshape并合并到之前的embeddings中（对于不同的seq，其对应的位置的position embedding是一样的，仅于位置相关）：  
position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)#[1, seq_length, 1*embedding_size]  
output += position_embeddings#[batch_size, seq_length, 1*embedding_size]，由于position_embeddings与batch_size无关。只跟位置有关。

3）LN和dropout  
output = layer_norm_and_dropout(output, dropout_prob)  
即：  
先执行LN，对最后一个维度进行normalization，即embedding_size的维度  
output_tensor = layer_norm(input_tensor, name)  
然后执行dropout，采取tf.nn.dropout函数，每个元素以rate概率会被置为0，如果没有置为0，则除以（1-rate），从而使得整体的期望sum不变。  
output_tensor = dropout(output_tensor, dropout_prob)  

后处理的返回值为：  
return output#[batch_size, seq_length, 1*embedding_size]，即后处理最终的输出结果  

## 2.3 encoder
encoder部分和transformer中的encoder结构几乎一样，函数为transformer_model()：    
```text
self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,#[batch_size, seq_length, 1*embedding_size]
            attention_mask=attention_mask,#[batch_size, seq_length, seq_length]
            hidden_size=config.hidden_size,#768 for base
            num_hidden_layers=config.num_hidden_layers,#12 for base
            num_attention_heads=config.num_attention_heads,#12 for base
            intermediate_size=config.intermediate_size,#3072，即feed forward的size
            intermediate_act_fn=get_activation(config.hidden_act),#feed forward的激活函数为gelu
            hidden_dropout_prob=config.hidden_dropout_prob,#0.1
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,#0.1
            initializer_range=config.initializer_range,#0.02
            do_return_all_layers=True)
```
1）几个重要的参数：  
attention_head_size = int(hidden_size / num_attention_heads)#64  
input_shape = get_shape_list(input_tensor, expected_rank=3)#[batch_size, seq_length, 1*embedding_size]  
batch_size = input_shape[0]#batch_size  
seq_length = input_shape[1]#seq_length  
input_width = input_shape[2]#1*embedding_size，等于hidden_size，即768  

2）两个子层  
对于每一层都包含两个子层（Sublayer）：Multi-Head Attention层和Position-wise Feed-Forward Networks层。并且每个子层都有residual connection和LN。  
可表示为LayerNorm(x + Sublayer(x))  
input_tensor#为之前后处理的结果，即最后embedding的结果  
prev_output = reshape_to_matrix(input_tensor)#nD to 2D，即[batch_size*seq_length, 1*embedding_size]  
layer_input = prev_output#[batch_size\*seq_length, 1*embedding_size]，可用于残差连接

3）Multi-Head Attention层（子层1）  
函数为attention_layer()
```text
attention_head = attention_layer(#[batch_size*seq_length, 12*64]
              from_tensor=layer_input,#[batch_size*seq_length, 1*embedding_size]
              to_tensor=layer_input,#[batch_size*seq_length, 1*embedding_size]
              attention_mask=attention_mask,#[batch_size, seq_length, seq_length]
              num_attention_heads=num_attention_heads,#12
              size_per_head=attention_head_size,#64
              attention_probs_dropout_prob=attention_probs_dropout_prob,#0.1
              initializer_range=initializer_range,#0.02
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
```
在attention_layer中进行了如下操作：  
- 首先通过对layer_input进行不同的线性变换分别得到query_layer，key_layer, value_layer。  
但是对于所有的query都是input与一样的矩阵进行点积进行线性变换得到的，key和value也是。  
如：
```text
query_layer = tf.layers.dense(#output为：[batch_size*seq_length, 12*64]
      from_tensor_2d,#[batch_size*seq_length, 1*embedding_size]
      num_attention_heads * size_per_head,#12*64
      activation=query_act,#query_act为None，即无激活函数，为线性变换
      name="query",
      kernel_initializer=create_initializer(initializer_range))
```
三者的shape均为：[batch_size*seq_length, 12*64]  
然后通过reshape和转置query_layer，key_layer, value_layer得到：  
[batch_size, 12, seq_length, 64]， 即：对于query_layer是：[B, N, F, H]，对于key_layer和value_layer是[B, N, T, H]  
即multi-heads    

- 然后QKV计算Scaled Dot-Product Attention；  
首先Q和K点积，并scale：  
transpose_b先转置为[batch_size, 12, 64, seq_length]  
attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)#[batch_size, 12, seq_length, seq_length]  
最后两个维度为：  
[[a11,a12,...a1m],  
 [a21,a22,...amm],  
 ...  
 [am1,am2,...amm]]  
attention_scores = tf.multiply(attention_scores,
                               1.0 / math.sqrt(float(size_per_head)))#[batch_size, 12, seq_length, seq_length]，即每个值除以8  
Scale的原因：  
We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by 1/math.sqrt(dk).  
即当dk较大时，Q，K之间的dot products，得到的值较大，导致softmax到达了一个梯度很小的区域，造成训练上的困难，因此需要scale。  


- 然后进行attention mask：
即无需attend to [PAD]，所以需要将[PAD]位置的attention score减小。  
先计算attention_mask，即：  
attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)  
[batch_size, seq_length, seq_length]，即最后一个维度存储的才是mask的值，包含0和1, [[1,1,1,...1,0,0,0]]*seq_length  
```text
[[1,1,1,...1,0,0,0],
 [1,1,1,...1,0,0,0],
 ...
 [1,1,1,...1,0,0,0]]
```
然后扩展维度，并将1减小为0,0减小为-10000：  
attention_mask = tf.expand_dims(attention_mask, axis=[1])#[batch_size, 1, seq_length, seq_length]  
```text
[[0,0,0,...,0,-10000,-10000,-10000],
 [0,0,0,...,0,-10000,-10000,-10000],
 ...
 [0,0,0,...,0,-10000,-10000,-10000]]
```
adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0#[batch_size, 1, seq_length, seq_length]  0->-10000, 1->0
相加，进行attention mask，相当于最后一个维度上[a11,a12,a13,...a1128] + [0,0,0,...-10000]  
attention_scores += adder#[batch_size, 12, seq_length, seq_length]  

- softmax:
attention_probs = tf.nn.softmax(attention_scores)#[batch_size, 12, seq_length, seq_length]， 即[B, N, F, T]

- dropout：
This is actually dropping out entire tokens to attend to, which might seem a bit unusual, but is taken from the original Transformer paper.
即对于有些应该attend to的tokens，也有可能被完全drop out，看似不合理，但原transformer就是这么做的。我猜这样可以增强噪音，增强模型的鲁棒性  
attention_probs = dropout(attention_probs, attention_probs_dropout_prob)#[batch_size, 12, seq_length, seq_length]， [B, N, F, T]

- matmul：
即同attention_probs（[B, N, F, T], [batch_size, 12, seq_length, seq_length]）与value_layer（[B, N, T, H],[batch_size, 12, seq_length, 64]）进行点积，得到新的context_layer：  
context_layer = tf.matmul(attention_probs, value_layer)#[batch_size, 12, seq_length, 64]， [B, N, F, H]
转置并reshape：  
context_layer = tf.transpose(context_layer, [0, 2, 1, 3])#[batch_size, seq_length, 12, 64]， [B, F, N, H]
context_layer = tf.reshape(context_layer,[batch_size * from_seq_length, num_attention_heads * size_per_head])#[batch_size*seq_length, 12*64]， [B*F, N*H]  
相当于Concat操作，concatenate multi-heads.  

因此attention_layer层最终返回的结果为：  
attention_output = context_layer#[batch_size*seq_length, 12*64]  
对attention_output进行处理：  
线性映射：  
attention_output = tf.layers.dense(attention_output, hidden_size, kernel_initializer=create_initializer(initializer_range))#[batch_size*seq_length, 768]  
然后dropout:  
attention_output = dropout(attention_output, hidden_dropout_prob)#[batch_size*seq_length, 768] 

4）Add & Norm 
然后Add（残差连接），并Norm（LN）：  
attention_output = layer_norm(attention_output + layer_input)#[batch_size*seq_length, 768]  

5）FFN（子层2）  
对应文章中的FFN(x) = max(0; xW1 + b1)W2 + b2，但原文是同的Relu激活函数，而bert用的是gelu。  
全连接层得到中间hidden layer([batch_size*seq_length, 3072])，即gelu(xW1 + b1)：    
```text
intermediate_output = tf.layers.dense(#[batch_size*seq_length, 3072]
    attention_output,#[batch_size*seq_length, 768]
    intermediate_size,#3072，即4*768
    activation=intermediate_act_fn,#gelu
    kernel_initializer=create_initializer(initializer_range))
```
然后再线性变换（Down-project），即xW2 + b2：  
```text
layer_output = tf.layers.dense(#无激活函数，即线性变换，输出[batch_size*seq_length, 768]
            intermediate_output,#[batch_size*seq_length, 3072]
            hidden_size,#768
            kernel_initializer=create_initializer(initializer_range))
```
然后dropout：  
layer_output = dropout(layer_output, hidden_dropout_prob)#[batch_size*seq_length, 768]

6）Add & Norm  
layer_output = layer_norm(layer_output + attention_output)#[batch_size*seq_length, 768]  

7)其他  
prev_output = layer_output#[batch_size*seq_length, 768]，即传递给下一层
all_layer_outputs.append(layer_output)#[[batch_size*seq_length, 768],...]，保存所有层的结果
final_outputs = []
for layer_output in all_layer_outputs:
  final_output = reshape_from_matrix(layer_output, input_shape)#batch_size, seq_length, 768]
  final_outputs.append(final_output)
即transformer_model（即BertModel().all_encoder_layer）最终的返回结果为：  
final_outputs#[[batch_size, seq_length, 768],...]  

## 2.4 output  
之前相当于得到self.all_encoder_layers = transformer_model()    
final hidden layer of encoder：  
self.sequence_output = self.all_encoder_layers[-1]#[batch_size, seq_length, 1*embedding_size]
pooler（[CLS]对应tensor进行处理）：  
first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)#即[CLS]的tensor，[batch_size, 1*embedding_size]  
tf.squeeze的作用：Removes dimensions of size 1 from the shape of a tensor.  
```text
self.pooled_output = tf.layers.dense(#[batch_size, 768]
    first_token_tensor,#[batch_size, 768]
    config.hidden_size,#768
    activation=tf.tanh,
    kernel_initializer=create_initializer(config.initializer_range))
```
“pooler”这个操作是针对segment-level (or segment-pair-level) classification tasks，where we need a fixed dimensional representation of the segment.  
即默认[CLS]位置的tensor将用于分类任务，并对该tensor接全连接层（tanh(xW+b)）。  

一般需要获取的内容：  
```text
  def get_pooled_output(self):
    return self.pooled_output#[batch_size, 768]

  def get_sequence_output(self):#[batch_size, seq_length, 768]
    """Gets final hidden layer of encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers#[[batch_size, seq_length, 768],...]

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).
    """
    return self.embedding_output#[batch_size, seq_length, 1*embedding_size]

  def get_embedding_table(self):
    return self.embedding_table#[vocab_size, embedding_size]
``` 

# 问答任务
## SQuAD 1.1
几乎没有对模型架构做出太大的调整，或者做数据增强；但是需要较为复杂的数据预处理和后处理，来解决(a)SQuAD文本长度可变性
（即通过切分passage）；(b)符号水平的回答注释，用于训练；
- 下载数据集到$BERT_BASE_DIR目录（SQuAD1.1）  
*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```shell
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

The dev set predictions will be saved into a file called `predictions.json`；

Evaluate：  
```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}  
You should see a result similar to the 88.5% reported in the paper for`BERT-Base`.  
实际结果为：  
{"exact_match": 80.63386944181646, "f1": 88.1236694661201}


另外先在TriviaQA上面微调，效果会更好：  
If you fine-tune for one epoch on
[TriviaQA](http://nlp.cs.washington.edu/triviaqa/) before this the results will
be even better, but you will need to convert TriviaQA into the SQuAD json
format.
## SQuAD 2.0
相比SQuAD 1.1，The SQuAD 2.0任务进行了拓展，增加了无答案的情况，更加贴近现实。  

# Using BERT to extract fixed feature vectors (like ELMo)
In certain cases, rather than fine-tuning the entire pre-trained model
end-to-end, it can be beneficial to obtained *pre-trained contextual
embeddings*, which are fixed contextual representations of each input token
generated from the hidden layers of the pre-trained model. This should also
mitigate most of the out-of-memory issues.

As an example, we include the script `extract_features.py` which can be used
like this:

```shell
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > input.txt
export BERT_BASE_DIR=/data/liubiao/PTMs/bert/models/uncased_L-12_H-768_A-12
python extract_features.py \
  --input_file=input.txt \
  --output_file=output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

This will create a JSON file (one line per line of input) containing the BERT
activations from each Transformer layer specified by `layers` (-1 is the final
hidden layer of the Transformer, etc.)  
即提取一句话的每个token对应的最后几层（FFN之后Add&Norm的vec）对应位置的vec。比如第一个token，提取的是最后1-4层，对应位置1的vec。  

# Tokenization
The basic procedure for sentence-level tasks is:（例子`run_classifier.py` and `extract_features.py`）  
1.  Instantiate an instance of `tokenizer = tokenization.FullTokenizer`
2.  Tokenize the raw text with `tokens = tokenizer.tokenize(raw_text)`.
3.  Truncate to the maximum sequence length. (You can use up to 512, but you
    probably want to use shorter if possible for memory and speed reasons.)
4.  Add the `[CLS]` and `[SEP]` tokens in the right place.

Word-level and span-level tasks (e.g., SQuAD and NER) are more complex, since
you need to maintain alignment between your input text and output text so that
you can project your training labels. SQuAD is a particularly complex example
because the input labels are *character*-based, and SQuAD paragraphs are often
longer than our maximum sequence length. See the code in `run_squad.py` to show
how we handle this.

Before we describe the general recipe for handling word-level tasks, it's
important to understand what exactly our tokenizer is doing. It has three main
steps:

1.  **Text normalization**: Convert all whitespace characters to spaces, and
    (for the `Uncased` model) lowercase the input and strip out accent markers.
    E.g., `John Johanson's, → john johanson's,`.

2.  **Punctuation splitting**: Split *all* punctuation characters on both sides
    (i.e., add whitespace around all punctuation characters). Punctuation
    characters are defined as (a) Anything with a `P*` Unicode class, (b) any
    non-letter/number/space ASCII character (e.g., characters like `$` which are
    technically not punctuation). E.g., `john johanson's, → john johanson ' s ,`

3.  **WordPiece tokenization**: Apply whitespace tokenization to the output of
    the above procedure, and apply
    [WordPiece](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py)
    tokenization to each token separately. (Our implementation is directly based
    on the one from `tensor2tensor`, which is linked). E.g., `john johanson ' s
    , → john johan ##son ' s ,`

The advantage of this scheme is that it is "compatible" with most existing
English tokenizers. For example, imagine that you have a part-of-speech tagging
task which looks like this:

```
Input:  John Johanson 's   house
Labels: NNP  NNP      POS NN
```

The tokenized output will look like this:

```
Tokens: john johan ##son ' s house
```

Crucially, this would be the same output as if the raw text were `John
Johanson's house` (with no space before the `'s`).

If you have a pre-tokenized representation with word-level annotations, you can
simply tokenize each input word independently, and deterministically maintain an
original-to-tokenized alignment:

```python
### Input
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
```

Now `orig_to_tok_map` can be used to project `labels` to the tokenized
representation.

There are common English tokenization schemes which will cause a slight mismatch
between how BERT was pre-trained. For example, if your input tokenization splits
off contractions like `do n't`, this will cause a mismatch. If it is possible to
do so, you should pre-process your data to convert these back to raw-looking
text, but if it's not possible, this mismatch is likely not a big deal.（即最好做单词还原，不然会有mismatch）

# sequence tag task
参见BERT-NER目录  
主要的细节：  
```text
这里遵循bert一般的套路，即最后一层dropout，再线性变换，再softmax；
model = modeling.BertModel()#bert模型
output_layer = model.get_sequence_output()#提取encoder的最后一层，[B, S, E]， [64, 64, 768]
output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)#训练时dropout，[64,64,768]
logits = hidden2tag(output_layer,num_labels)#即接一个全连接层，无激活函数；[64,64,10]
logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])#reshape，[64,64,10]
loss,predict  = softmax_layer(logits, labels, num_labels, mask)#[64, 64, 10], [64, 64]
在计算loss的时候，需要去掉[PAD]处的loss；
```
需要注意的是原paper中提到：We use the representation of the first sub-token as the input to the token-level classifier over the NER label set。  
也就是说多于包含多个sub-token的token，将第一个sub-token的向量作为分类器的输入，那么也就是说后面的sub-token应该是不参与loss计算的（推测）。  
序列标注任务对应pytorch_transformers的BertForTokenClassification类，即token classification。该类源码为：
```text
self.classifier = nn.Linear(config.hidden_size, config.num_labels)
outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
sequence_output = outputs[0]# sequence_output, pooled_output, (hidden_states), (attentions)
sequence_output = self.dropout(sequence_output)#即最后一层hidden layer，经过dropout
logits = self.classifier(sequence_output)#线性变换
```
即与tf代码逻辑是完全一样的。  

# Pre-training
- 数据：  
两个语料库：  
the BooksCorpus (800M words)   
English Wikipedia (2,500M words)：只提取了文章，忽略了 lists, tables, and headers；  
数据下载（见bert github）  

- 数据处理
得到plain text file，one sentence per line。document之间由一个空行分隔。示例见sample_text.txt。  

- 生成tfrecord
```shell
export BERT_BASE_DIR=/data/liubiao/PTMs/bert/pretrain_uncased_L-12_H-768_A-12
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```
得到的样本示例为：  
```text
INFO:tensorflow:tokens: [CLS] this text is included to make sure unicode is handled bracelet : 鍔� 鍔� [MASK] 鍖� 鍖� 岽� ##岽� ##岬� ##岬� ##唳� [MASK] ##唳� ##唳� ##唳� ##唳� [SEP] text should be one - [MASK] - per [MASK] line , wes [MASK] documents . [SEP]
INFO:tensorflow:input_ids: 101 2023 3793 2003 2443 2000 2191 2469 27260 2003 8971 19688 1024 1778 1779 103 1781 1782 1493 30030 30031 30032 29893 103 29895 29896 29897 29898 102 3793 2323 2022 2028 1011 103 1011 2566 103 2240 1010 2007 4064 3210 103 5491 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:masked_lm_positions: 11 15 23 34 37 38 43 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:masked_lm_ids: 7919 1780 29894 6251 1011 2240 2090 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:masked_lm_weights: 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
INFO:tensorflow:next_sentence_labels: 0
```
原文为：  
This text is included to make sure Unicode is handled properly: 力加勝北区ᴵᴺᵀᵃছজটডণত  
Text should be one-sentence-per-line, with empty lines between documents.  
即位置11的properly替换为bracelet；位置38的line还是line；  

即：无论是替换为[MASK]、还是替换为随机词，还是没变的词，都需要预测为原token，计算loss。  
对于NSP任务，NSP为True的情况下，sentence A是一个document连续的sentences连在一起得到的，而sentence B则是紧接在A后面的连续sentences连在一起的；
NSP为False的情况，A还是如此，但B则变成了其他document中随机取的连续sentences连在一起得到的。阳性样本和阴性样本采样的概率是一样的，即0.5。  

参数`max_predictions_per_seq` is the maximum number of masked LM predictions per
sequence. You should set this to around `max_seq_length` * `masked_lm_prob` (the
script doesn't do that automatically because the exact value needs to be passed
to both scripts).

pretrain：  
```shell
export BERT_BASE_DIR=/data/liubiao/PTMs/bert/pretrain_uncased_L-12_H-768_A-12
python run_pretraining.py \
  --input_file=tf_examples.tfrecord \
  --output_dir=pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```
源码：
```text
(masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss
```
即loss为mlm和nsp两个loss的和。


从头pretrain不要init_checkpoint参数，如果是多次pretrain，后续的训练则需要init_checkpoint参数。  
pretrain结果（官方）：
```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```
pretrain结果（实际）：
```text
INFO:tensorflow:***** Eval results *****
I0830 16:06:01.350611 140338225133312 run_pretraining.py:483] ***** Eval results *****
INFO:tensorflow:  global_step = 20
I0830 16:06:01.350720 140338225133312 run_pretraining.py:485]   global_step = 20
INFO:tensorflow:  loss = 9.251094
I0830 16:06:01.350955 140338225133312 run_pretraining.py:485]   loss = 9.251094
INFO:tensorflow:  masked_lm_accuracy = 0.06743769
I0830 16:06:01.351062 140338225133312 run_pretraining.py:485]   masked_lm_accuracy = 0.06743769
INFO:tensorflow:  masked_lm_loss = 8.5400095
I0830 16:06:01.351148 140338225133312 run_pretraining.py:485]   masked_lm_loss = 8.5400095
INFO:tensorflow:  next_sentence_accuracy = 0.53625
I0830 16:06:01.351230 140338225133312 run_pretraining.py:485]   next_sentence_accuracy = 0.53625
INFO:tensorflow:  next_sentence_loss = 0.7045548
I0830 16:06:01.351311 140338225133312 run_pretraining.py:485]   next_sentence_loss = 0.7045548
```

## Pre-training tips and caveats
*   **If using your own vocabulary, make sure to change `vocab_size` in
    `bert_config.json`. If you use a larger vocabulary without changing this,
    you will likely get NaNs when training on GPU or TPU due to unchecked
    out-of-bounds access.**
*   If your task has a large domain-specific corpus available (e.g., "movie
    reviews" or "scientific papers"), it will likely be beneficial to run
    additional steps of pre-training on your corpus, starting from the BERT
    checkpoint.
*   The learning rate we used in the paper was 1e-4. However, if you are doing
    additional steps of pre-training starting from an existing BERT checkpoint,
    you should use a smaller learning rate (e.g., 2e-5).
*   Current BERT models are English-only, but we do plan to release a
    multilingual model which has been pre-trained on a lot of languages in the
    near future (hopefully by the end of November 2018).
*   Longer sequences are disproportionately expensive because attention is
    quadratic to the sequence length. In other words, a batch of 64 sequences of
    length 512 is much more expensive than a batch of 256 sequences of
    length 128. The fully-connected/convolutional cost is the same, but the
    attention cost is far greater for the 512-length sequences. Therefore, one
    good recipe is to pre-train for, say, 90,000 steps with a sequence length of
    128 and then for 10,000 additional steps with a sequence length of 512. The
    very long sequences are mostly needed to learn positional embeddings, which
    can be learned fairly quickly. Note that this does require generating the
    data twice with different values of `max_seq_length`.
*   If you are pre-training from scratch, be prepared that pre-training is
    computationally expensive, especially on GPUs. If you are pre-training from
    scratch, our recommended recipe is to pre-train a `BERT-Base` on a single
    [preemptible Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing), which
    takes about 2 weeks at a cost of about $500 USD (based on the pricing in
    October 2018). You will have to scale down the batch size when only training
    on a single Cloud TPU, compared to what was used in the paper. It is
    recommended to use the largest batch size that fits into TPU memory.


## Learning a new WordPiece vocabulary
This repository does not include code for *learning* a new WordPiece vocabulary.
The reason is that the code used in the paper was implemented in C++ with
dependencies on Google's internal libraries. For English, it is almost always
better to just start with our vocabulary and pre-trained models. For learning
vocabularies of other languages, there are a number of open source options
available. However, keep in mind that these are not compatible with our
`tokenization.py` library:

*   [Google's SentencePiece library](https://github.com/google/sentencepiece)

*   [tensor2tensor's WordPiece generation script](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)

*   [Rico Sennrich's Byte Pair Encoding library](https://github.com/rsennrich/subword-nmt)


# 其他注意事项
1. For classification tasks, the first vector (corresponding to [CLS]) is
used as as the "sentence vector". Note that this only makes sense because
the entire model is fine-tuned.
2. Whole Word Masking Models
2019年BERT增加了BERT-WWM模型，本质上是mask的时候，之前的随机mask sub-word，但是现在一个word如果被分成了几个sub-word，那么要么都不mask，要么都mask。
因为之前的情况下，只有部分sub-word被mask的情况下，很容易预测出被mask的sub-word，现在都mask了，难度会提升，也会更多的依赖上下文。
即pre-processing部分进行了修改。原始的方案是randomly select WordPiece tokens to mask. For example:

`Input Text: the man jumped up , put his basket on phil ##am ##mon ' s head`
`Original Masked Input: [MASK] man [MASK] up , put his [MASK] on phil
[MASK] ##mon ' s head`

The new technique is called Whole Word Masking. In this case, we always mask
*all* of the the tokens corresponding to a word at once. The overall masking
rate remains the same.

`Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK]
[MASK] ' s head`

训练不变，还是预测each masked WordPiece token independently，
This can be enabled during data generation by passing the flag
`--do_whole_word_mask=True` to `create_pretraining_data.py`.
WWM模型：  
*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
实验比较：  
Model                                    | SQUAD 1.1 F1/EM | Multi NLI Accuracy
---------------------------------------- | :-------------: | :----------------:
BERT-Large, Uncased (Original)           | 91.0/84.3       | 86.05
BERT-Large, Uncased (Whole Word Masking) | 92.8/86.7       | 87.07
BERT-Large, Cased (Original)             | 91.5/84.8       | 86.09
BERT-Large, Cased (Whole Word Masking)   | 92.9/86.7       | 86.46
效果确实有所提升。
