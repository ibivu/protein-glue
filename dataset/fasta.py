import Bio.SeqIO
import numpy as np

def read_fasta(path):
    recs = Bio.SeqIO.parse(path, 'fasta')

    for rec in recs:
        a = np.array(rec.seq, 'c').view(np.uint8).astype(np.int32) - 65
        yield (rec.name, rec.description, a)


def to_seq_array(seqs):
    max_seq_len = max(len(seq) for seq in seqs)

    pad_seqs = []
    for seq in seqs:
        pad_seq = np.pad(seq + 4, [[0, max_seq_len - seq.shape[0]]])

        pad_seqs.append(pad_seq)

    return np.stack(pad_seqs, axis=0)
