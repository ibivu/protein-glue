#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Bio.SeqIO
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import collections
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import gzip
import os

currentPath = os.path.dirname(os.path.abspath(__file__))

#open pfam database
f = gzip.open(os.path.join(currentPath, 'Pfam-A.fasta.gz'), 'rt', encoding='latin1')

all_families = set()
family_counter = collections.Counter()
family_total_lengths = collections.Counter()
family_to_seqs = collections.defaultdict(list)
family_to_big_seqs = collections.defaultdict(list)
for i, rec in enumerate(Bio.SeqIO.parse(f, 'fasta')):
    length = len(rec.seq)
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

selected_seqs = []
for seqs in family_to_seqs.values():
    selected_seqs.extend(seqs)
random.shuffle(selected_seqs)

selected_big_seqs = []
for seqs in family_to_big_seqs.values():
    selected_big_seqs.extend(seqs)
random.shuffle(selected_big_seqs)

#determine ratio's w.r.t whole dataset
ratio_train = 0.8
ratio_val = 0.1
ratio_test = 0.1

#split of testset
selected_seqs_remaining, selected_seqs_test = train_test_split(selected_seqs, test_size=ratio_test, shuffle=True, random_state=0)
selected_big_remaining, selected_big_test = train_test_split(selected_big_seqs, test_size=ratio_test, shuffle=True, random_state=0)

#adjust validation ratio w.r.t remaining dataset
ratio_remaining = 1 - ratio_test
ratio_val_adjusted = ratio_val / ratio_remaining

#split trainingset in validation and testset
selected_seqs_training, selected_seq_validation = train_test_split(selected_seqs_remaining, test_size = ratio_val_adjusted, shuffle=True, random_state=0)
selected_big_training, selected_big_validation = train_test_split(selected_big_remaining, test_size = ratio_val_adjusted, shuffle=True, random_state=0)

def _bytes_feature(value):
    b = value.encode('ascii')
    a = np.frombuffer(b, dtype=np.uint8, count=len(b)) - 61 #smaller number -> more extra special tokens
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[a.tobytes()]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_to_file(filename, data):
    with tf.io.TFRecordWriter(filename) as fo:
        for seq in data:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'sequence': _bytes_feature(seq),
            }))

            fo.write(ex.SerializeToString())

def write_seqs(name, seqs, K=10):
    chunk_size = len(seqs) // K
    for k, i in enumerate(range(0, len(seqs), chunk_size)):
        fname = "{0}-k{1}.tfrecord".format(name, k+1)
        print("Writing {0}:{1} to {2}...".format(i, i + chunk_size, fname))
        write_to_file('./output_datasplit_traintestval/' + fname, seqs[i:i + chunk_size])

#create directory to store files if does not yet exist
if not os.path.exists('output_datasplit_traintestval'):
    os.makedirs('output_datasplit_traintestval')

#store training, test and validation sets normal sequences in file
write_seqs('bert-sequences-testset', selected_seqs_test,1) #testset
write_seqs('bert-sequences', selected_seqs_training,9) #training
write_seqs('bert-sequences-validation', selected_seq_validation, 1) #validation


#store trainign, test and validation sets BIG sequences in file
write_seqs('bert-sequences-big-testset', selected_big_test,1) #test
write_seqs('bert-sequences-big', selected_big_training, 1) #training
write_seqs('bert-sequences-big-validation', selected_big_validation, 1) #validation
