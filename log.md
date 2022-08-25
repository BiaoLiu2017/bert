# Flows
##1. download & env & model  
git clone git@github.com:google-research/bert.git  
pip install tensorflow-gpu==1.15.0  
download models(links in bert github page)  
注：numpy最好是1.19.5版本，版本过高可能导致报错；  

##2. download fine-tuning data  
bert-large need too large GPU RAM to reproduce paper。
so fine-tuning bert-base.
download glue data(execute the blow code)(有的网络可能受限导致无法下载，在本地windows下下载glue data的):
https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3

##3. fine-tuning  
按照readme中的fine-tuning代码就能正常的fine-tuning了（注意修改bert路径以及glue_data路径）。

##4. bert文本预处理  
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

##5. 生成InputExample
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

##6.写入tfrecord
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
##7. Train  
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
