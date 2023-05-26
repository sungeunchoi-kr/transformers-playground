#
# Toy transformer decoder that simply outputs "I am groot".
# Contains training code.
# 
# Most of the code is a copy/modification of the tutorial:
# - NLP Demystified 15: Transformers From Scratch + Pre-training and Transfer Learning With BERT/GPT
#   - https://www.youtube.com/watch?v=acxqoltilME
#   - https://colab.research.google.com/github/nitinpunjabi/nlp-demystified/blob/main/notebooks/nlpdemystified_seq2seq_and_attention.ipynb#scrollTo=2tYGhcB2zAAN
#   - https://colab.research.google.com/github/nitinpunjabi/nlp-demystified/blob/main/notebooks/nlpdemystified_transformers_and_pretraining.ipynb#scrollTo=BV_fyVfIPzjH

import io
import json
import numpy as np
import random
import re
import tensorflow as tf
import unicodedata
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from bpemb import BPEmb

# d - the size of the embedding vector
# t - the # of tokens (also called sequence length)
def scaled_dot_product_attention(query, key, value, mask=None):
    # get the embedding vector size
    key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
    # print("*** key=")
    # print(key)

    # print("tf.shape(key)=")
    # print(tf.shape(key))

    # print("*** key_dim=")
    # print(key_dim)

    # applying the formula.
    # how to imagine this: 
    # [ q1      [ k1  k2  .. kt
    #   q2   *     |   |      |
    #   ..         |   |      |
    #   qd ]                    ]
    # where the key is already transposed.
    #     and q1, ... k1, ... represent a single vector -- a "word" (an embedding, to be more exact.)
    # 
    # now, for matrix multiplication -- for one row si in the resulting matrix S (for score),
    # it is the result of "fixing" the word qi and running it across all the `key` words in the
    # sentence. thus, a row in S tells you the score for each key for a given query.
    scaled_scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(key_dim)

    if mask is not None:
        scaled_scores = tf.where(mask==0, -np.inf, scaled_scores)

    softmax = tf.keras.layers.Softmax()
    weights = softmax(scaled_scores) 

    # weights:
    # [
    #   s1  (list of scores per "word" when queried by k1)
    #   s2
    #   ..
    #   st
    # ]
    # t X t matrix.
    #
    # value:
    # [ v1
    #   v2
    #   ..
    #   vt ]
    # t x d matrix.
    # 
    # result of matmul: N x d matrix.

    return tf.matmul(weights, value), weights


class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
  
        self.d_head = self.d_model // self.num_heads
  
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
  
        # Linear layer to generate the final output.
        self.dense = tf.keras.layers.Dense(self.d_model)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
  
        split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        return tf.transpose(split_inputs, perm=[0, 2, 1, 3])
    
    def merge_heads(self, x):
        batch_size = x.shape[0]
  
        merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(merged_inputs, (batch_size, -1, self.d_model))
  
    def call(self, q, k, v, mask):
        qs = self.wq(q)
        ks = self.wk(k)
        vs = self.wv(v)
  
        qs = self.split_heads(qs)
        ks = self.split_heads(ks)
        vs = self.split_heads(vs)
  
        output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
        output = self.merge_heads(output)
  
        return self.dense(output), attn_weights


def feed_forward_network(d_model, hidden_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.mhsa1 = MultiHeadSelfAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, hidden_dim)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()
  
    # Note the decoder block takes two masks. One for the first MHSA, another
    # for the second MHSA.
    def call(self, target, training, decoder_mask):
        mhsa_output1, attn_weights = self.mhsa1(target, target, target, decoder_mask)
        mhsa_output1 = self.dropout1(mhsa_output1, training=training)
        mhsa_output1 = self.layernorm1(mhsa_output1 + target)

        ffn_output = self.ffn(mhsa_output1)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(ffn_output + mhsa_output1)

        return output, attn_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.pos_embed = tf.keras.layers.Embedding(max_seq_len, self.d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.blocks = [DecoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)]

    def call(self, target, training, decoder_mask):
        token_embeds = self.token_embed(target)

        # Generate position indices.
        num_pos = target.shape[0] * self.max_seq_len
        pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
        pos_idx = np.reshape(pos_idx, target.shape)

        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds, training=training)

        for block in self.blocks:
            x, weights = block(x, training, decoder_mask)

        return x, weights


class DecoderTransformer(tf.keras.Model):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim,
                 target_vocab_size, max_target_len, dropout_rate=0.1):
        super(DecoderTransformer, self).__init__()

        self.decoder = Decoder(num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                               max_target_len, dropout_rate)
      
        # The final dense layer to generate logits from the decoder output.
        self.output_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, target_input_seqs, training, decoder_mask):
        decoder_output, decoder_attn_weights = self.decoder(target_input_seqs, training,
                                                            decoder_mask)

        return self.output_layer(decoder_output), decoder_attn_weights


train_preprocessed_target = [ "I am Groot" for _ in range(100) ]

def tag_target_sentences(sentences):
    tagged_sentences = map(lambda s: (' ').join(['<sos>', s, '<eos>']), sentences)
    return list(tagged_sentences)

train_tagged_preprocessed_target = tag_target_sentences(train_preprocessed_target)

target_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')
target_tokenizer.fit_on_texts(train_tagged_preprocessed_target)
target_tokenizer.get_config()

target_vocab_size = len(target_tokenizer.word_index) + 1
print(target_vocab_size)

def generate_decoder_inputs_targets(sentences, tokenizer):
    seqs = tokenizer.texts_to_sequences(sentences)
    decoder_inputs = [s[:-1] for s in seqs] # Drop the last token in the sentence.
    decoder_targets = [s[1:] for s in seqs] # Drop the first token in the sentence.

    return decoder_inputs, decoder_targets

train_decoder_inputs, train_decoder_targets = generate_decoder_inputs_targets(train_tagged_preprocessed_target, 
                                                                              target_tokenizer)

print(train_decoder_inputs[:3], train_decoder_targets[:3])

print(target_tokenizer.sequences_to_texts(train_decoder_inputs[:3]), 
      target_tokenizer.sequences_to_texts(train_decoder_targets[:3]))

max_decoding_len = len(max(train_decoder_inputs, key=len))
print(max_decoding_len)

target_input_seqs = train_decoder_inputs

padded_target_input_seqs = tf.keras.preprocessing.sequence.pad_sequences(target_input_seqs, padding="post")
padded_train_decoder_inputs = padded_target_input_seqs
# print('padded_target_input_seqs:')
# print(padded_target_input_seqs)
# print('')

padded_train_decoder_targets = tf.keras.preprocessing.sequence.pad_sequences(train_decoder_targets, padding="post")

dec_padding_mask = tf.cast(tf.math.not_equal(padded_target_input_seqs, 0), tf.float32)
dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]
# print('dec_padding_mask:')
# print(dec_padding_mask)
# print('')

target_input_seq_len = padded_target_input_seqs.shape[1]
look_ahead_mask = tf.linalg.band_part(tf.ones((target_input_seq_len, 
                                               target_input_seq_len)), -1, 0)
# print('look_ahead_mask:')
# print(look_ahead_mask)
# print('')

dec_mask = tf.minimum(dec_padding_mask, look_ahead_mask)
# print("The decoder mask:")
# print(dec_mask)

d_model=4
num_heads=1

transformer = DecoderTransformer(
    num_blocks = 2,
    d_model = d_model,
    num_heads = num_heads,
    hidden_dim = 48,
    target_vocab_size = target_vocab_size,
    max_target_len = padded_target_input_seqs.shape[1])

transformer_output, _ = transformer(padded_target_input_seqs,
                                    True, 
                                    dec_mask)

print(transformer_output)
print(f"Transformer output before training {transformer_output.shape}:")
 
transformer_output_idx = tf.argmax(transformer_output, axis=-1)

print(transformer_output_idx[0])
print(target_tokenizer.sequences_to_texts(transformer_output_idx[0:1].numpy()))


# ==== training ==== #


# This custom loss function is simply a wrapper around a sparse categorical crossentropy loss,
# but with a mask of 1s and 0s. Any target element of 0 (i.e. a padding value) will get a mask
# value of 0, everything else will get a mask of 1, and only target values corresponding to a
# mask value of 1 will be used for loss calculation.
def loss_func(targets, logits):
    print(" -- loss_func called.")
    # print(targets)
    # print(logits)
  
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    mask = tf.cast(tf.math.not_equal(targets, 0), tf.float32)

    return ce_loss(targets, logits, sample_weight=mask)



class TransformerTrainer(tf.keras.Model):
    def __init__(self, decoder):
        super(TransformerTrainer, self).__init__()

        self.decoder = decoder

    # This method will be called by model.fit for each batch.
    @tf.function
    def train_step(self, inputs):
        print(' -- train_step() called.')
        loss = 0.

        decoder_input_seq, decoder_target_seq = inputs
        
        with tf.GradientTape() as tape:    
            # We need to create a loop to iterate through the target sequences
            # print("loop length: " + str(decoder_target_seq.shape[1]))

            # for i in range(decoder_target_seq.shape[1]):

            dec_padding_mask = tf.cast(tf.math.not_equal(decoder_input_seq, 0), tf.float32)
            dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

            target_input_seq_len = decoder_target_seq.shape[1]
            look_ahead_mask = tf.linalg.band_part(tf.ones((target_input_seq_len, 
                                                          target_input_seq_len)), -1, 0)
            dec_mask = tf.minimum(dec_padding_mask, look_ahead_mask)

            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension (just like in the previous example).
            # next_decoder_input = tf.expand_dims(decoder_input_seq[:, i], 1)
            logits, _ = self.decoder(decoder_input_seq, True, dec_mask)

            # print("logits:")
            # print(logits)
            # print("decoder_target_seq")
            # print(decoder_target_seq)

            # The loss is now accumulated through the whole batch
            loss += self.loss(decoder_target_seq, logits)

        # Update the parameters and the optimizer
        variables = self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'loss': loss / decoder_target_seq.shape[1]}


optimizer = tf.keras.optimizers.Adam()

trainer = TransformerTrainer(transformer)
trainer.compile(optimizer=optimizer, loss=loss_func)

dataset = tf.data.Dataset.from_tensor_slices((padded_train_decoder_inputs, 
                                              padded_train_decoder_targets)).batch(batch_size=5, drop_remainder=True)
print(dataset)

epochs = 12
trainer.fit(dataset, epochs=epochs)

padded_target_input_seqs_first_element = tf.expand_dims( padded_target_input_seqs[0], axis=0 )
print(padded_target_input_seqs_first_element)

seq = target_tokenizer.texts_to_sequences(["<sos> I"])
seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=4, padding="post")
print(seq)

transformer_output, decoder_attn_weights = transformer(seq,
                                                       True, 
                                                       dec_mask)

# print(transformer_output)
print(f"Transformer output {transformer_output.shape}:")
print("decoder_attn_weights:")
print(decoder_attn_weights)

transformer_output_idx = tf.argmax(transformer_output, axis=-1)

print(transformer_output_idx[0])
print(target_tokenizer.sequences_to_texts(transformer_output_idx[0:10].numpy()))
