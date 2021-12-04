from os import sep
import tensorflow as tf
import numpy as np
from hashlib import md5

import constants as c

def create_reduced_alphabet(num_special_tokens=4):
    base_indices = np.array(
        [1, 5, 0, 5, 5, 4, 1, 7, 0, 0, 6, 0, 0, 5, 0, 3, 5, 6, 2, 2, 0, 0, 4, -num_special_tokens, 4, 5],
        dtype=np.int32
    )
    indices = base_indices + num_special_tokens

    special_tokens = np.arange(num_special_tokens, dtype=np.int32)

    reduced_alphabet = np.concatenate([special_tokens, indices])

    return reduced_alphabet

REDUCED_ALPHABET_MAPPING = create_reduced_alphabet(c.NUM_SPECIAL_SYMBOLS)

def _parse_examples(target_reduced_alphabet=False, max_sequence_length=128):
    def inner(examples):
        features = {
            "sequence": tf.io.FixedLenFeature((), tf.string, default_value=""),
            "is_next": tf.io.FixedLenFeature((), tf.string, default_value=""),
            "segment_label": tf.io.FixedLenFeature((), tf.string, default_value=""),
        }
        parsed_features = tf.io.parse_example(examples, features)
        seq = tf.io.decode_raw(parsed_features["sequence"], tf.dtypes.uint8, fixed_length=max_sequence_length)
        seq = tf.cast(seq, dtype=tf.int32)

        is_next = tf.io.decode_raw(parsed_features["is_next"], tf.dtypes.uint8, fixed_length=1)
        is_next = tf.cast(is_next[:, 0], dtype=tf.int32)

        segment_label = tf.io.decode_raw(parsed_features["segment_label"], tf.dtypes.uint8, fixed_length=max_sequence_length)
        segment_label = tf.cast(segment_label, dtype=tf.int32)

        # create two versions of the string, one with the start of sequence token, used for input
        # and one used as an output target, with an end of sequence token.
        input_seq = tf.identity(seq)
        target_seq = tf.identity(seq)

        # because the sequence is relatively non-conserved we have an option to collapse
        # amino acid types down to composite amino acids with roughly the same properties
        # this makes the prediction task easier, and in theory more focused on the structural
        # relationships rather than getting the exact amino acid type correct.
        if target_reduced_alphabet:
            target_seq = tf.gather(REDUCED_ALPHABET_MAPPING, target_seq)

        return (input_seq, target_seq, is_next, segment_label)

    return inner

def create_dataset(filenames, batch_size=32, target_reduced_alphabet=False, max_sequence_length=128):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        _parse_examples(target_reduced_alphabet=target_reduced_alphabet, max_sequence_length=max_sequence_length),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def dataset_iter(ds, ds_big, optimizer, num_steps_big_batch_max_proportion):
    it = iter(ds)
    it_big = iter(ds_big) if ds_big else None

    batch = 1
    epoch = 1
    is_big = False  # If we want to seperate the big sequences from the rest

    chance_big_batch = 0
    hash_value = 0

    while True:

        cur_step = optimizer.iterations.numpy() + 1

        if num_steps_big_batch_max_proportion:
            if int(cur_step) >= int(num_steps_big_batch_max_proportion):
                chance_big_batch = 0.5
            else:
                cur_percent = round((int(cur_step) / int(num_steps_big_batch_max_proportion)), 2)
                chance_big_batch = cur_percent * 0.5

            h = md5()
            h.update(str(cur_step).encode('ascii'))
            hash_value = int.from_bytes(h.digest()[:5], 'big')

        if (((hash_value % 100) < (chance_big_batch * 100)) or (((batch % 10) == 5) and not num_steps_big_batch_max_proportion)) and ds_big:
            # print('Yielding big batch')
            try:
                inp, tar, is_next, segment_label = next(it_big)
            except StopIteration:
                it_big = iter(ds_big)
                inp, tar, is_next, segment_label = next(it_big)
            is_big = True
        else:
            try:
                inp, tar, is_next, segment_label = next(it)
            except StopIteration:
                epoch += 1
                batch = 1
                it = iter(ds)
                inp, tar, is_next, segment_label = next(it)
            is_big = False

        yield epoch, batch, is_big, inp, tar, is_next, segment_label
        batch += 1

def val_dataset_iter(ds, ds_big):
    it = iter(ds)
    it_big = iter(ds_big) if ds_big else None

    batch = 1
    batch_big = 1
    is_big = False  # If we want to seperate the big sequences from the rest

    while True:
        if ds_big:
            try:
                inp, tar, is_next, segment_label = next(it_big)
                is_big = True
            except StopIteration:
                ds_big = None
        else:
            try:
                inp, tar, is_next, segment_label = next(it)
                is_big = False
            except StopIteration:
                break

        yield is_big, batch, batch_big, inp, tar, is_next, segment_label
        if is_big:
            batch_big +=1
        else:
            batch += 1

def subtrain_dataset_iter(ds, ds_big, selected_batches, selected_batches_big):
    it = iter(ds)
    it_big = iter(ds_big) if ds_big else None

    batch = 1
    batch_big = 1
    is_big = False  # If we want to seperate the big sequences from the rest

    while True:
        if ds_big:
            try:
                inp, tar, is_next, segment_label = next(it_big)
                is_big = True

                if batch_big in selected_batches_big:
                    yield is_big, batch, batch_big, inp, tar, is_next, segment_label

                batch_big += 1
            except StopIteration:
                ds_big = None
        else:
            try:
                inp, tar, is_next, segment_label = next(it)
                is_big = False
                
                if batch in selected_batches:
                    yield is_big, batch, batch_big, inp, tar, is_next, segment_label
                    
                batch += 1
            except StopIteration:
                break
