import tensorflow as tf
import tensorflow_addons as tfa

class ASAClassifier(tf.keras.Model):
    def __init__(self, dff, rate=0.1):
        super(ASAClassifier, self).__init__()

        self.linear_layer = tf.keras.layers.Dense(dff, activation=tfa.activations.gelu) # (batch_size, seq_len, dff)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.regression_layer = tf.keras.layers.Dense(1) # Regression task, so linear output

    def call(self, input, bert_output, training):
        x = tf.concat([tf.one_hot(input, 30), bert_output], axis=-1)
        x = self.linear_layer(bert_output)
        x = self.dropout(x, training=training)
        x = self.regression_layer(x)
        x = tf.squeeze(x, axis=2)

        return x
