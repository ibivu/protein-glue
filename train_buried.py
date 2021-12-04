import tensorflow as tf
import time
import click
from tensorflow_addons.optimizers import LAMB

from dataset.bur import create_dataset
from model import BERTTransformer, BURClassifier

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_function(real, pred, loss_weights):
    loss_ = loss_object(real, pred)
    loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)
    loss_ *= loss_weights
    return tf.reduce_sum(loss_)/tf.reduce_sum(loss_weights)

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

POSITIONAL_ENCODING_MAX_LENGTH = 3800
LOGGING_EVERY_STEPS = 10

@click.command()
@click.argument('input_files', nargs=-1)
@click.argument('output_dir', nargs=1)
@click.option('--validation-file', default=None)
@click.option('--learning-rate', default=1e-4)
@click.option('--pretrain-checkpoint-dir', default=None)
@click.option('--num-layers', default=12)
@click.option('--num-heads', default=8)
@click.option('--d-ff', default=-1)
@click.option('--d-model', default=768)
@click.option('--dropout-rate', default=0.1)
@click.option('--batch-size', default=32)
@click.option('--keep-checkpoints', default=2)
@click.option('--num-batches-checkpoint', default=250)
@click.option('--num-epochs', default=5)
@click.option('--num-steps', default=1000000)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
@click.option('--mixed-float/--no-mixed-float', default=False)
@click.option('--freeze-pretrained/--no-freeze-pretrained', default=False)
@click.option('--tensorboard-dir', default=None)
def main(learning_rate, num_layers, num_heads, d_ff, d_model, dropout_rate, batch_size, keep_checkpoints, pretrain_checkpoint_dir,
         num_batches_checkpoint, num_epochs, num_steps, mixed_float, reduced_target_alphabet, validation_file, input_files, output_dir,
         freeze_pretrained, tensorboard_dir):
    if mixed_float:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if d_ff == -1:
        d_ff = d_model * 4
    input_vocab_size = 30
    target_vocab_size = 12 if reduced_target_alphabet else 30

    if tensorboard_dir:
        print("Creating Tensorboard")
        print(tensorboard_dir)
        writer = tf.summary.create_file_writer(tensorboard_dir)
    else:
        None

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        pad_mask = tf.math.logical_not(tf.math.equal(inp, 0))
        loss_weights = tf.cast(pad_mask, tf.float32)

        with tf.GradientTape() as tape:
            _, activations = transformer(inp, True, True)
            activations = activations[:, :, -1, :]
            predictions = classifier(inp, activations, True)

            loss = loss_function(tar, predictions, loss_weights)
            if mixed_float:
                loss = optimizer.get_scaled_loss(loss)

        trainable_variables = list(classifier.trainable_variables)
        if not freeze_pretrained:
            trainable_variables.extend(transformer.trainable_variables)

        gradients = tape.gradient(loss, trainable_variables)
        if mixed_float:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    @tf.function(input_signature=train_step_signature)
    def accuracy_step(inp, tar):
        _, activations = transformer(inp, False, True)
        activations = activations[:, :, -1, :]
        predictions = classifier(inp, activations, False)

        loss_weights = tf.cast(tf.math.logical_not(tf.math.equal(inp, 0)), tf.float32)
        # Don't include the padded or non-prediction positions in the accuracy.
        #train_accuracy(tar, predictions, sample_weight=loss_weights)

        #do not update accuracy here because 2 accuracy metrics and if statement not possible
        return tar, predictions, loss_weights

    seq_trainging = sum(1 for _ in tf.data.TFRecordDataset(input_files))
    seq_validation = sum(1 for _ in tf.data.TFRecordDataset(validation_file))
    print("number of sequences training: {}".format(seq_trainging))
    print("number of sequences validation: {}".format(seq_validation))

    transformer = BERTTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=d_ff,
        inp_vocab_size=input_vocab_size,
        tar_vocab_size=target_vocab_size,
        pe=POSITIONAL_ENCODING_MAX_LENGTH,
        rate=dropout_rate
    )
    classifier = BURClassifier(dff=d_ff, rate=dropout_rate)

    if pretrain_checkpoint_dir:
        pretrain_ckpt = tf.train.Checkpoint(transformer=transformer)
        pretrain_ckpt.restore(tf.train.latest_checkpoint(pretrain_checkpoint_dir)).expect_partial()
        print("Loaded pre-trained model from checkpoint!")
        print(pretrain_checkpoint_dir)

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(learning_rate, num_steps, end_learning_rate=0.0, power=1.0)
    optimizer = LAMB(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay_rate=0.01)
    if mixed_float:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    ds = create_dataset(input_files, batch_size=batch_size, max_length=512)
    validation_ds = create_dataset([validation_file], batch_size=batch_size, max_length=512) if validation_file else None

    ckpt = tf.train.Checkpoint(transformer=transformer,
                            classifier=classifier,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=keep_checkpoints)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!')

#################################################################################
    cur_step = None
    start = time.time()
    for epoch in range(num_epochs):
        batch = 1
        start_batch = time.time()
        validation_iter = iter(validation_ds) if validation_ds else None

        for (inp, tar) in ds:
            cur_step = optimizer.iterations.numpy() + 1

            if batch % LOGGING_EVERY_STEPS == 0:
                now = time.time()
                seqs_per_sec = (batch_size * LOGGING_EVERY_STEPS) / (now - start_batch)
                start_batch = now

                try:
                    accuracy_inp, accuracy_tar = next(validation_iter) if validation_iter else (inp, tar)
                except StopIteration:
                    validation_iter = iter(validation_ds) if validation_ds else None
                    accuracy_inp, accuracy_tar = next(validation_iter) if validation_iter else (inp, tar)

                targets, predictions, weights = accuracy_step(accuracy_inp, accuracy_tar)
                train_accuracy(targets, predictions, sample_weight = weights)

                print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} Accuracy {:.4f}'.format(
                    cur_step, epoch + 1, batch, seqs_per_sec, train_accuracy.result()))

                if writer:
                    with writer.as_default():
                        tf.summary.scalar('accuracy', train_accuracy.result(), cur_step)

                train_accuracy.reset_states()
            else:
                train_step(inp, tar)


            if cur_step % num_batches_checkpoint == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {}, batch {} at {}'.format(epoch + 1, batch, ckpt_save_path))

            batch += 1

            if cur_step > num_steps:
                break
        if cur_step > num_steps:
            break

################################################################################
    #do a whole validation
    for (inp_val, tar_val) in validation_ds:
        targets_val, predictions_val, weights_val = accuracy_step(inp_val, tar_val)
        val_accuracy.update_state(targets_val, predictions_val, sample_weight = weights_val)

    print("Performance over validation: Accuracy {:.4f}".format(val_accuracy.result()))

    print ('Time taken training: {} secs\n'.format(time.time() - start))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
