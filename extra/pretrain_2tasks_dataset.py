#!/usr/bin/env python
# coding: utf-8
import gzip
import os
import sys

import Bio.SeqIO
import random
import tensorflow as tf
import collections
import numpy as np

from sklearn.model_selection import train_test_split

NUM_SPECIAL_TOKENS = 20

#determine ratio's w.r.t whole dataset
RATIO_TRAIN = 0.8
RATIO_VAL = 0.1
RATIO_TEST = 0.1

#open pfam database
f = gzip.open(sys.argv[1], 'rt', encoding='latin1')

all_families = set()
family_counter = collections.Counter()
family_total_lengths = collections.Counter()
family_to_seqs = collections.defaultdict(list)
family_to_big_seqs = collections.defaultdict(list)
for i, rec in enumerate(Bio.SeqIO.parse(f, 'fasta')):
    length = len(rec.seq) + 3  # 3 tokens for CLS and 2 x SEP
    # Exclude peptides and extremely long domains
    if 50 <= length <= 512:
        name = rec.description
        words = name.split()
        family = words[2].split(';')[0].split('.')[0]

        family_counter[family] += 1
        family_total_lengths[family] += length
        all_families.add(family)

        if length <= 128:
            family_to_seqs[family].append(str(rec.seq))
        elif length <= 512:
            family_to_big_seqs[family].append(str(rec.seq))

        # if i > 1000:
        #     break

selected_seqs = []
for seqs in family_to_seqs.values():
    selected_seqs.extend(seqs)
random.shuffle(selected_seqs)

selected_big_seqs = []
for seqs in family_to_big_seqs.values():
    selected_big_seqs.extend(seqs)
random.shuffle(selected_big_seqs)

#split off testset
selected_seqs_remaining, selected_seqs_test = train_test_split(selected_seqs, test_size=RATIO_TEST, shuffle=True, random_state=0)
selected_big_remaining, selected_big_test = train_test_split(selected_big_seqs, test_size=RATIO_TEST, shuffle=True, random_state=0)

#adjust validation ratio w.r.t remaining dataset
ratio_remaining = 1 - RATIO_TEST
ratio_val_adjusted = RATIO_VAL / ratio_remaining

#split trainingset in validation and testset
selected_seqs_training, selected_seq_validation = train_test_split(selected_seqs_remaining, test_size = ratio_val_adjusted, shuffle=True, random_state=0)
selected_big_training, selected_big_validation = train_test_split(selected_big_remaining, test_size = ratio_val_adjusted, shuffle=True, random_state=0)

def _encode(seq):
    b = seq.encode('ascii')
    return np.frombuffer(b, dtype=np.uint8, count=len(b)) - ord('A') + NUM_SPECIAL_TOKENS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

PAD_TOKEN = np.array([0], dtype=np.uint8)
CLS_TOKEN = np.array([1], dtype=np.uint8)
SEP_TOKEN = np.array([2], dtype=np.uint8)

def write_to_file(filename, data):
    with tf.io.TFRecordWriter(filename) as fo:
        for index, seq in enumerate(data):

            # Encode string into ascii
            encoded_seq = _encode(seq)

            # Determine where to cut sequence
            cutting_ratio = random.uniform(0.4, 0.6)
            cutting_index = int(round(encoded_seq.shape[0] * cutting_ratio))

            first_half = encoded_seq[:cutting_index]
            remainder = encoded_seq.shape[0] - cutting_index

            if random.random() < 0.5:
                is_next = np.array([0], dtype=np.uint8)  # False

                # Get random sequence index not equal to current sequence
                rand_index = index
                while rand_index == index or len(data[rand_index]) <= remainder:
                    rand_index = random.randint(0, len(data) - 1)

                encoded_seq2 = _encode(data[rand_index])
                second_half = np.flipud(encoded_seq2[:remainder])  # Amino acids from random sequence with length=remainder in reverse order
            else:
                is_next = np.array([1], dtype=np.uint8)  # True

                second_half = encoded_seq[cutting_index:]

            new_seq = np.concatenate([CLS_TOKEN, first_half, SEP_TOKEN, second_half, SEP_TOKEN])

            segment_first_half = np.ones_like(first_half)
            segment_second_half = np.ones_like(second_half)
            segment_second_half[:] = 2

            segment_label = np.concatenate((PAD_TOKEN, segment_first_half, PAD_TOKEN, segment_second_half, PAD_TOKEN))

            ex = tf.train.Example(features=tf.train.Features(feature={
                'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[new_seq.tobytes()])),  #_bytes_feature(seq),
                'is_next': tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_next.tobytes()])),
                'segment_label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[segment_label.tobytes()])),
            }))

            fo.write(ex.SerializeToString())

def write_seqs(name, seqs, K=10):
    chunk_size = len(seqs) // K
    for k, i in enumerate(range(0, len(seqs), chunk_size)):
        fname = "{0}-k{1}.tfrecord".format(name, k+1)
        print("Writing {0}:{1} to {2}...".format(i, i + chunk_size, fname))
        write_to_file('./pretrain_data/' + fname, seqs[i:i + chunk_size])

#create directory to store files if does not yet exist
if not os.path.exists('pretrain_data'):
    os.makedirs('pretrain_data')

#store training, test and validation sets normal sequences in file
write_seqs('bert-sequences-testset', selected_seqs_test,1) #testset
write_seqs('bert-sequences', selected_seqs_training,9) #training
write_seqs('bert-sequences-validation', selected_seq_validation, 1) #validation


#store trainign, test and validation sets BIG sequences in file
write_seqs('bert-sequences-big-testset', selected_big_test,1) #test
write_seqs('bert-sequences-big', selected_big_training, 1) #training
write_seqs('bert-sequences-big-validation', selected_big_validation, 1) #validation
