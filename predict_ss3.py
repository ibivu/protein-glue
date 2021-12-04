import tensorflow as tf
import numpy as np
import click

from dataset.fasta import read_fasta, to_seq_array
from model import BERTTransformer, SS3Classifier

predict_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

IDX_TO_Q3_SYMBOL = {0: 'C', 1: 'H', 2: 'E'}

def blocks(s, size=78):
    return [s[i:i+size] for i in range(0, len(s), size)]

@click.command()
@click.argument('checkpoint_dir')
@click.argument('input_fasta')
@click.argument('output_file')
@click.option('--num-layers', default=12)
@click.option('--num-heads', default=8)
@click.option('--d-ff', default=-1)
@click.option('--d-model', default=768)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
def main(num_layers, num_heads, d_ff, d_model, reduced_target_alphabet, input_fasta, checkpoint_dir, output_file):
    @tf.function(input_signature=predict_step_signature)
    def predict_step(inp):
        _, activations = transformer(inp, False, True)
        activations = activations[:, :, -1, :]
        predictions = classifier(inp, activations, False)

        return predictions

    if d_ff == -1:
        d_ff = d_model * 4
    input_vocab_size = 30
    target_vocab_size = 12 if reduced_target_alphabet else 30

    transformer = BERTTransformer(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        target_vocab_size,
        pe=513,
        rate=0.0,
    )
    classifier = SS3Classifier(dff=d_ff, rate=0.0)

    ckpt = tf.train.Checkpoint(transformer=transformer, classifier=classifier)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    seqs = []
    for name, _, seq in read_fasta(input_fasta):
        inp = tf.constant(to_seq_array([seq]))
        pred = predict_step(inp)
        pred = tf.argmax(pred[:, :, 1:], axis=2).numpy().flatten()

        seq = "".join([IDX_TO_Q3_SYMBOL[x] for x in pred])
        seqs.append((name, seq))

    with open(output_file, 'w') as fo:
        for name, seq in seqs:
            fo.write(f'>{name}\n')
            for block in blocks(seq):
                fo.write(f'{block}\n')

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()