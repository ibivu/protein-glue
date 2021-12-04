import tensorflow as tf
import numpy as np
import click

from dataset.fasta import read_fasta, to_seq_array
from model import BERTTransformer

predict_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@click.command()
@click.argument('checkpoint_dir')
@click.argument('input_fasta')
@click.argument('output_file')
@click.option('--num-layers', default=12)
@click.option('--num-heads', default=8)
@click.option('--d-ff', default=-1)
@click.option('--d-model', default=768)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
@click.option('--embed-num-layers', default=4)
def main(num_layers, num_heads, d_ff, d_model, reduced_target_alphabet, embed_num_layers, input_fasta, checkpoint_dir, output_file):
    @tf.function(input_signature=predict_step_signature)
    def predict_step(inp):
        predictions, activations = transformer(inp, False, True)
        predictions = tf.argmax(predictions, axis=2)

        return predictions, activations

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

    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    outputs = []

    for name, _, seq in read_fasta(input_fasta):
        inp = tf.constant(to_seq_array([seq]))
        pred, emb = predict_step(inp)
        pred = tf.cast(pred, tf.int32)

        emb_arr = tf.squeeze(emb, 0).numpy()
        emb_arr = emb_arr[:, -embed_num_layers:, :]
        emb_arr = np.reshape(emb_arr, (emb_arr.shape[0], emb_arr.shape[1] * emb_arr.shape[2]))

        outputs.append((name, emb_arr))

    np.save(output_file, np.array(outputs, dtype=object))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()