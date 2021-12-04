import tensorflow as tf
import tensorflow_addons as tfa

class PNIClassifier(tf.keras.Model):
    def __init__(self, dff, rate=0.1):
        super(PNIClassifier, self).__init__()

        self.nonlinear_layer = tf.keras.layers.Dense(dff, activation=tfa.activations.gelu)  # (batch_size, seq_len, dff)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.classify_layer = tf.keras.layers.Dense(3)  #2 classes
        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)

    def call(self, input, bert_output, training):
        x = tf.concat([tf.one_hot(input, 30), bert_output], axis=-1)
        x = self.nonlinear_layer(bert_output)
        x = self.dropout(x, training=training)
        x = self.classify_layer(x)
        x = self.probabilities(x)

        return x
