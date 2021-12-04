import tensorflow as tf

from .transformer import Encoder, create_padding_mask

import constants as c

class BERTTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, inp_vocab_size, tar_vocab_size, pe, rate=0.1):
        super(BERTTransformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                inp_vocab_size, pe, rate)

    def call(self, inp, inp_extra, training, return_activations=False):
        padding_mask = create_padding_mask(inp)
        output, activations = self.encoder(inp, inp_extra, training, padding_mask, return_activations)  # (batch_size, inp_seq_len, d_model)

        if return_activations:
            return output, activations
        else:
            return output

class NSPEmbeddingLayer(tf.keras.Model):
    def __init__(self, d_model):
        super(NSPEmbeddingLayer, self).__init__()

        self.segment_embedding = tf.keras.layers.Embedding(3, d_model)  # 3 because either segment 1 or 2 or special token / padding

    def call(self, segment_label):
        return self.segment_embedding(segment_label)

class FinalLayer(tf.keras.Model):
    def __init__(self, tar_vocab_size):
        super(FinalLayer, self).__init__()

        self.final_layer = tf.keras.layers.Dense(tar_vocab_size)
        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)

    def call(self, embeddings):
        final_output = self.final_layer(embeddings)  # (batch_size, inp_seq_len, tar_vocab_size)
        probs_output = self.probabilities(final_output)

        return probs_output

class NSPLayer(tf.keras.Model):
    def __init__(self):
        super(NSPLayer, self).__init__()

        self.final_layer = tf.keras.layers.Dense(2)
        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)

    def call(self, embeddings):
        final_output = self.final_layer(embeddings[:, 0, :])  # (batch_size, inp_seq_len, 2)
        probs_output = self.probabilities(final_output)

        return probs_output

def create_pad_mask(inp):
    return tf.math.logical_not(tf.math.less(inp, c.NUM_SPECIAL_SYMBOLS)) # currently 20 special token types in use

def create_prediction_mask(inp, input_vocab_size, predict_rate, predict_random_rate, predict_nop_rate):
    # Prediction mask is a boolean mask of the positions we will be predicting.
    # We try to stay as close to the BERT article as possible, including the masking:
    # * 1.0 - predict_random_rate - predict_nop_rate will be replaced by a mask token.
    # * predict_nop_rate will retain their original value
    # * predict_random_rate will be set to a randomized value that is very likely not to be the true value,
    #   which is an addition in our case as our small vocabulary size makes the chance of randomizing
    #   to the old (and correct) token a non-negligible boost in model performance
    prediction_mask = tf.math.less_equal(tf.random.uniform(tf.shape(inp)), predict_rate)
    r = tf.random.uniform(tf.shape(prediction_mask))

    inp0 = tf.identity(inp)

    randomize_mask = tf.math.less_equal(r, predict_random_rate)
    mask_token_mask = tf.math.greater(r, predict_random_rate + predict_nop_rate)
    pad_mask = create_pad_mask(inp)
    pad_prediction_mask = tf.logical_and(prediction_mask, pad_mask)

    # Scatter update to set the non-padding, prediction positions to the mask token value
    mask_token_mask_idxs = tf.where(tf.logical_and(pad_prediction_mask, mask_token_mask))
    mask_values = tf.fill((tf.shape(mask_token_mask_idxs)[0],), 3)
    inp = tf.tensor_scatter_nd_update(inp, mask_token_mask_idxs, mask_values)

    # Do the same for the randomize mask, but set them to a random amino acid
    randomize_pad_prediction_mask = tf.logical_and(pad_prediction_mask, randomize_mask)
    randomize_mask_idxs = tf.where(randomize_pad_prediction_mask)
    mask_values = tf.random.uniform((tf.shape(randomize_mask_idxs)[0],), c.NUM_SPECIAL_SYMBOLS, input_vocab_size, dtype=tf.int32)
    inp = tf.tensor_scatter_nd_update(inp, randomize_mask_idxs, mask_values)

    # Do two rounds of checking where the randomized value happens to be equal to the original and rerandomize.
    rerandomize_r1_mask_idxs = tf.where(tf.logical_and(randomize_pad_prediction_mask, tf.math.equal(inp, inp0)))
    mask_values = tf.random.uniform((tf.shape(rerandomize_r1_mask_idxs)[0],), c.NUM_SPECIAL_SYMBOLS, input_vocab_size, dtype=tf.int32)
    inp = tf.tensor_scatter_nd_update(inp, rerandomize_r1_mask_idxs, mask_values)
    rerandomize_r2_mask_idxs = tf.where(tf.logical_and(randomize_pad_prediction_mask, tf.math.equal(inp, inp0)))
    mask_values = tf.random.uniform((tf.shape(rerandomize_r2_mask_idxs)[0],), c.NUM_SPECIAL_SYMBOLS, input_vocab_size, dtype=tf.int32)
    inp = tf.tensor_scatter_nd_update(inp, rerandomize_r2_mask_idxs, mask_values)

    return inp, prediction_mask, pad_prediction_mask