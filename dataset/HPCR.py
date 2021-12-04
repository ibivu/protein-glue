from numpy import float32
import tensorflow as tf
import constants as c

def _parse_example(example_proto):
    features = {
        "sequence": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "hydrophobic_patch": tf.io.VarLenFeature(tf.float32),
        "hp_class": tf.io.VarLenFeature(tf.int64),
        "hp_rank": tf.io.VarLenFeature(tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    input_seq = tf.io.decode_raw(parsed_features["sequence"], tf.dtypes.uint8)

    # We get raw ASCII bytes from the tensorflow file format, shift their values so 'A' maps to index 3,
    # because we reserve 0 for padding / masked values, 1 for the start of sequence marker, and 2 for
    # the end of sequence marker

    """
    {'B', 'U', "'", 'O', 'X'} occur 2268 times
    ': geef 0
    X: any amino acid; give 0
    U: similar to C
    B: give 0
    O: give 0
    ASCII codes:    U: 85
                    B: 66
                    ': 39
                    O: 79
                    X: 88
    Example: replace all values in list a for a_max if higher than a_max
    a = tf.where(tf.less_equal(a, a_max), a, a_max)

    To give it 0, give it 61 before subtracting 61.
    Do it only for ' others will be mapped (Maurits)
    """
    #input_seq = tf.where(tf.not_equal(input_seq, 85), input_seq, 61)
    #input_seq = tf.where(tf.not_equal(input_seq, 66), input_seq, 61)
    #input_seq = tf.where(tf.not_equal(input_seq, 79), input_seq, 61)
    #input_seq = tf.where(tf.not_equal(input_seq, 88), input_seq, 61)
    input_seq = tf.where(tf.not_equal(input_seq, 39), input_seq, 61)

    input_seq = input_seq - 65 + c.NUM_SPECIAL_SYMBOLS
    input_seq = tf.cast(input_seq, tf.int32)

    segment_label = tf.ones_like(input_seq, dtype=tf.float32)

    #saw 116 ranks per protein. Set t0 128 to be sure? and to
    target_seq = tf.sparse.to_dense(parsed_features['hp_rank'],
                                            default_value=0)
    #such that not existing can be 1
    target_seq = target_seq + 1
    target_seq = tf.cast(target_seq, tf.int32)

    return (input_seq, target_seq, segment_label)

def create_dataset_hpcr(filenames, batch_size=16, max_length=128):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(2000, reshuffle_each_iteration=True) #2000 is half of number of training sequences
    dataset = dataset.map(_parse_example)
    if max_length:
        dataset = dataset.filter(lambda x, y, z: tf.shape(x)[0] <= max_length)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None])).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
