from numpy import float32
import tensorflow as tf
import constants as c

def _parse_example(example_proto):
    features = {
        "sequence": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "solvent_accessibility": tf.io.VarLenFeature(tf.float32),
        "buried": tf.io.VarLenFeature(tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    input_seq = tf.io.decode_raw(parsed_features["sequence"], tf.dtypes.uint8)

    # We get raw ASCII bytes from the tensorflow file format, shift their values so 'A' maps to index 3,
    # because we reserve 0 for padding / masked values, 1 for the start of sequence marker, and 2 for
    # the end of sequence marker

    #NOTE: for solvent accessibility the default value should be -1 and the others could be zero
    input_seq = input_seq -  65 + c.NUM_SPECIAL_SYMBOLS
    input_seq = tf.cast(input_seq, tf.int32)

    segment_label = tf.ones_like(input_seq, dtype=tf.float32)

    target_seq = tf.sparse.to_dense(parsed_features['solvent_accessibility'],
                                            default_value=-1)
    target_seq = target_seq + 1
    #target_seq = tf.cast(target_seq, tf.int32)
    # target_seq = target_seq * 100
    # target_seq = tf.cast(target_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.float32)

    return (input_seq, target_seq, segment_label)

def create_dataset_asa(filenames, batch_size=16, max_length=128):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_example)
    if max_length:
        dataset = dataset.filter(lambda x, y, z: tf.shape(x)[0] <= max_length)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None])).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
