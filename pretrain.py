import tensorflow as tf
import time
import random
import signal
import click
from tensorflow.python.ops.variables import trainable_variables
from tensorflow_addons.optimizers import LAMB

from dataset.seqs import create_dataset, dataset_iter, val_dataset_iter, subtrain_dataset_iter
from model import BERTTransformer, create_prediction_mask, FinalLayer, NSPLayer, NSPEmbeddingLayer
import constants as c

kill_signal = False

def signal_handler(sig, frame):
    global kill_signal

    print('Kill signal caught: setting global flag for main loop to terminate')
    kill_signal = True

def mask_loss_function(loss_object):
    def inner(real, pred, loss_weights, batch_size):
        loss_ = loss_object(real, pred)
        loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)
        batch_size = tf.cast(batch_size, dtype=loss_.dtype)

        return tf.nn.compute_average_loss(loss_, sample_weight=loss_weights, global_batch_size=batch_size)
    return inner

def nsp_loss_function(loss_object):
    def inner(real, pred, batch_size):
        loss_ = loss_object(real, pred)
        batch_size = tf.cast(batch_size, dtype=loss_.dtype)

        return tf.nn.compute_average_loss(loss_, global_batch_size=batch_size)
    return inner

def get_metrics():
    train_mask_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_mask_loss')
    train_mask_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_mask_accuracy')
    train_nsp_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_nsp_loss')
    train_nsp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_nsp_accuracy')

    return (train_mask_loss, train_mask_accuracy, train_nsp_loss, train_nsp_accuracy)

def get_datasets(input_files, big_input_file, val_files, big_val_file, batch_size, big_batch_size, strategy, reduced_target_alphabet):
    ds = create_dataset(
        input_files,
        batch_size=batch_size,
        target_reduced_alphabet=reduced_target_alphabet,
        max_sequence_length=128,
    )
    ds = strategy.experimental_distribute_dataset(ds)

    ds_big = None
    if big_input_file:
        ds_big = create_dataset(
            [big_input_file],
            batch_size=big_batch_size,
            target_reduced_alphabet=reduced_target_alphabet,
            max_sequence_length=512,
        )
        ds_big = strategy.experimental_distribute_dataset(ds_big)

    val_ds = None
    val_ds_big = None
    if val_files:
        val_ds = create_dataset(
            val_files,
            batch_size=batch_size,
            target_reduced_alphabet=reduced_target_alphabet,
            max_sequence_length=128,
        )
        val_ds = strategy.experimental_distribute_dataset(val_ds)
    if big_val_file:
        val_ds_big = create_dataset(
            [big_val_file],
            batch_size=big_batch_size,
            target_reduced_alphabet=reduced_target_alphabet,
            max_sequence_length=512,
        )
        val_ds_big = strategy.experimental_distribute_dataset(val_ds_big)

    return (ds, ds_big, val_ds, val_ds_big)

POSITIONAL_ENCODING_MAX_LENGTH = 520
LOGGING_EVERY_STEPS = 1000

@click.command()
@click.argument('input_files', nargs=-1)
@click.argument('output_dir', nargs=1)
@click.option('--target', type=click.Choice(['standard', 'tpu', 'gpu', 'gpu_mixed']), default='standard')
@click.option('--learning-rate', default=1e-4)
@click.option('--lr-reduction-patience', default=4)
@click.option('--lr-reduction-factor', default=0.5)
@click.option('--lr-reduction-threshold', default=0.0001)
@click.option('--num-layers', default=12)
@click.option('--num-heads', default=8)
@click.option('--d-ff', default=-1)
@click.option('--d-model', default=768)
@click.option('--dropout-rate', default=0.1)
@click.option('--batch-size', default=32)
@click.option('--keep-checkpoints', default=2)
@click.option('--num-batches-checkpoint', default=5000)
@click.option('--num-epochs', default=5)
@click.option('--num-steps', default=1000000)
@click.option('--num-steps-big-batch-max-proportion', default=None)
@click.option('--big-input-file', default=None)
@click.option('--big-batch-size', default=4)
@click.option('--predict-rate', default=0.15)
@click.option('--predict-random-rate', default=0.1)
@click.option('--predict-nop-rate', default=0.1)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
@click.option('--val-files', default=None)
@click.option('--big-val-file', default=None)
@click.option('--tensorboard-dir', default=None)
def main(target, learning_rate, lr_reduction_patience, lr_reduction_factor, lr_reduction_threshold, num_layers, num_heads, d_ff, d_model, dropout_rate, batch_size, keep_checkpoints,
         num_batches_checkpoint, num_epochs, num_steps, num_steps_big_batch_max_proportion, big_input_file, big_batch_size, predict_rate, predict_random_rate, predict_nop_rate,
         reduced_target_alphabet, input_files, output_dir, val_files, big_val_file, tensorboard_dir):

    strategy = tf.distribute.get_strategy()
    if target in {'gpu', 'gpu_mixed'}:
        strategy = tf.distribute.MirroredStrategy()
    elif target == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    if target == 'gpu_mixed':
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    elif target == 'tpu':
        tf.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')

    if d_ff == -1:
        d_ff = d_model * c.D_FF_D_MODEL_RATIO
    input_vocab_size = 26 + c.NUM_SPECIAL_SYMBOLS
    target_vocab_size = 8 + c.NUM_SPECIAL_SYMBOLS if reduced_target_alphabet else input_vocab_size

    def step(inp, tar, is_next, segment_label, is_training):
        inp, _, pad_prediction_mask = create_prediction_mask(
            inp,
            input_vocab_size=input_vocab_size,
            predict_rate=predict_rate,
            predict_nop_rate=predict_nop_rate,
            predict_random_rate=predict_random_rate
        )
        loss_weights = tf.cast(pad_prediction_mask, tf.float32)

        with tf.GradientTape() as tape:
            inp_extra = nsp_embedding(segment_label)
            embeddings = transformer(inp, inp_extra, is_training)
            mask_predictions = mask_predictor(embeddings)
            masked_token_loss = mask_loss_fn(tar, mask_predictions, loss_weights, tf.shape(inp)[0])
            nsp_prediction = nsp_predictor(embeddings)
            nsp_loss = nsp_loss_fn(is_next, nsp_prediction, tf.shape(inp)[0])
            loss = masked_token_loss + nsp_loss

            if target in {'tpu', 'gpu_mixed'}:
                loss = optimizer.get_scaled_loss(loss)

        if is_training:
            train_vars = transformer.trainable_variables + nsp_embedding.trainable_variables + mask_predictor.trainable_variables + nsp_predictor.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            if target in {'tpu', 'gpu_mixed'}:
                gradients = optimizer.get_unscaled_gradients(gradients)
            optimizer.apply_gradients(zip(gradients, train_vars))

        # Don't include the padded or non-prediction positions in the accuracy.
        mask = tf.cast(pad_prediction_mask, dtype=tf.float32)

        train_mask_loss(tar, mask_predictions, sample_weight=mask)
        train_mask_accuracy(tar, mask_predictions, sample_weight=mask)
        train_nsp_loss(is_next, nsp_prediction)
        train_nsp_accuracy(is_next, nsp_prediction)

        return loss

    @tf.function
    def distributed_step(inp, tar, is_next, segment_label, is_training):
        per_replica_losses = strategy.run(step, args=(inp, tar, is_next, segment_label, is_training))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)

    with strategy.scope():
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
        nsp_embedding = NSPEmbeddingLayer(d_model)
        nsp_predictor = NSPLayer()
        mask_predictor = FinalLayer(target_vocab_size)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        mask_loss_fn = mask_loss_function(loss_object)
        nsp_loss_fn = nsp_loss_function(loss_object)

        # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(learning_rate, num_steps, end_learning_rate=0.0, power=1.0)
        optimizer = LAMB(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay_rate=0.01)
        if target in {'tpu', 'gpu_mixed'}:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # Needs to be created inside the strategy scope according to some of the examples
        ckpt = tf.train.Checkpoint(
            transformer=transformer,
            mask_predictor=mask_predictor,
            nsp_predictor=nsp_predictor,
            nsp_embedding=nsp_embedding,
            optimizer=optimizer
        )

        train_mask_loss, train_mask_accuracy, train_nsp_loss, train_nsp_accuracy = get_metrics()

    # Used by tensorboard to write loss values to
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir) if tensorboard_dir else None

    ds, ds_big, val_ds, val_ds_big = get_datasets(
        input_files, big_input_file, val_files, big_val_file,
        batch_size, big_batch_size, strategy, reduced_target_alphabet
    )

    num_batches_train = 0
    if big_input_file:
        num_batches_train_big = 0

    for element in ds:
        num_batches_train += 1
    if big_input_file:
        for element in ds_big:
            num_batches_train_big += 1

    print("Number of batches in training dataset: %s" % num_batches_train)
    if big_input_file:
        print("Number of batches in big training dataset: %s" % num_batches_train_big)

    ckpt_manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=keep_checkpoints)

    # if checkpoint(s) exists, restore the latest one
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored')

    start = time.time()
    start_batch = time.time()
    seqs_batch = 0

    cur_epoch = 1

    prev_val_loss = 99999
    no_improvement = 0

    print ('Starting training...')
    for epoch, batch, is_big, inp, tar, is_next, segment_label in dataset_iter(ds, ds_big, optimizer, num_steps_big_batch_max_proportion):
        if epoch > num_epochs:
            break

        if kill_signal:
            print('Job was killed! Stop training...')

            print ('Saving final checkpoint for epoch {}, batch {} at {}'.format(epoch, batch, ckpt_save_path))
            ckpt_save_path = ckpt_manager.save()

            break

        if epoch != cur_epoch:
            if val_ds and val_ds_big:
                print ('Starting validation...')
                for val_is_big, val_batch, val_batch_big, val_inp, val_tar, val_is_next, val_segment_label in val_dataset_iter(val_ds, val_ds_big):

                    distributed_step(val_inp, val_tar, val_is_next, val_segment_label, False)

                print ('Validation averages (Epoch {}) | Mask loss {:.4f} accuracy {:.4f} | NSP loss {:.4f} accuracy {:.4f}'.format(
                    epoch - 1,
                    train_mask_loss.result(),
                    train_mask_accuracy.result(),
                    train_nsp_loss.result(),
                    train_nsp_accuracy.result(),
                ))

                # ReduceLROnPlateau
                if train_mask_loss.result() >= prev_val_loss - lr_reduction_threshold:
                    no_improvement += 1
                else:
                    no_improvement = 0

                prev_val_loss = train_mask_loss.result()
                current_learning_rate = optimizer.__getattribute__('learning_rate')

                if no_improvement == lr_reduction_patience:
                    new_learning_rate = current_learning_rate * lr_reduction_factor
                    print ('Learning rate reduced from {} to {}'.format(current_learning_rate, new_learning_rate))
                    optimizer.__setattr__('learning_rate', new_learning_rate)
                    current_learning_rate = new_learning_rate

                if tensorboard_writer:
                    with tensorboard_writer.as_default():
                        # Capture variables (as functions of cur_step) for tensorboard
                        tf.summary.scalar('val_mask_loss', train_mask_loss.result(), cur_step)
                        tf.summary.scalar('val_mask_accuracy', train_mask_accuracy.result(), cur_step)
                        tf.summary.scalar('val_nsp_loss', train_nsp_loss.result(), cur_step)
                        tf.summary.scalar('val_nsp_accuracy', train_nsp_accuracy.result(), cur_step)
                        tf.summary.scalar('learning_rate', current_learning_rate, cur_step)

                train_mask_loss.reset_states()
                train_mask_accuracy.reset_states()
                train_nsp_loss.reset_states()
                train_nsp_accuracy.reset_states()

                # ckpt_save_path = ckpt_manager.save()
                # print ('Saving checkpoint after validation for epoch {} at {}'.format(epoch, ckpt_save_path))

                print ('Starting training subset evaluation...')
                # Set up subset of trainingset for validation process
                rand_batches_train = [s for s in range(1, num_batches_train)]
                selected_batches_train = random.sample(rand_batches_train, val_batch)
                rand_batches_train_big = [s for s in range(1, num_batches_train_big)]
                selected_batches_train_big = random.sample(rand_batches_train_big, val_batch_big)

                selected_batches_train.sort()
                selected_batches_train_big.sort()

                for subtrain_is_big, subtrain_batch, subtrain_batch_big, subtrain_inp, subtrain_tar, subtrain_is_next, subtrain_segment_label in subtrain_dataset_iter(ds, ds_big, selected_batches_train, selected_batches_train_big):

                    distributed_step(subtrain_inp, subtrain_tar, subtrain_is_next, subtrain_segment_label, False)

                print ('Subtrain averages (Epoch {}) | Mask loss {:.4f} accuracy {:.4f} | NSP loss {:.4f} accuracy {:.4f}'.format(
                    epoch - 1,
                    train_mask_loss.result(),
                    train_mask_accuracy.result(),
                    train_nsp_loss.result(),
                    train_nsp_accuracy.result(),
                ))

                if tensorboard_writer:
                    with tensorboard_writer.as_default():
                        # Capture variables (as functions of cur_step) for tensorboard
                        tf.summary.scalar('subtrain_mask_loss', train_mask_loss.result(), cur_step)
                        tf.summary.scalar('subtrain_mask_accuracy', train_mask_accuracy.result(), cur_step)
                        tf.summary.scalar('subtrain_nsp_loss', train_nsp_loss.result(), cur_step)
                        tf.summary.scalar('subtrain_nsp_accuracy', train_nsp_accuracy.result(), cur_step)

                train_mask_loss.reset_states()
                train_mask_accuracy.reset_states()
                train_nsp_loss.reset_states()
                train_nsp_accuracy.reset_states()

            cur_epoch = epoch

        cur_step = optimizer.iterations.numpy() + 1
        seqs_batch += batch_size

        distributed_step(inp, tar, is_next, segment_label, True)

        if batch % LOGGING_EVERY_STEPS == 0:
            now = time.time()
            seqs_per_sec = seqs_batch / (now - start_batch)
            start_batch = now
            seqs_batch = 0

            print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} | Mask loss {:.4f} accuracy {:.4f} | NSP loss {:.4f} accuracy {:.4f}'.format(
                cur_step,
                epoch,
                batch,
                seqs_per_sec,
                train_mask_loss.result(),
                train_mask_accuracy.result(),
                train_nsp_loss.result(),
                train_nsp_accuracy.result(),
            ))

        if tensorboard_writer:
            with tensorboard_writer.as_default():
                # Capture variables (as functions of cur_step) for tensorboard
                tf.summary.scalar('mask_loss', train_mask_loss.result(), cur_step)
                tf.summary.scalar('mask_accuracy', train_mask_accuracy.result(), cur_step)
                tf.summary.scalar('nsp_loss', train_nsp_loss.result(), cur_step)
                tf.summary.scalar('nsp_accuracy', train_nsp_accuracy.result(), cur_step)
                # Plot when big batch was processed for debugging
                if num_steps_big_batch_max_proportion:
                    if is_big:
                        tf.summary.scalar('binary_plot_big', 1, cur_step)
                    else:
                        tf.summary.scalar('binary_plot_big', 0, cur_step)

        # Reset so we don't get running averages
        train_mask_loss.reset_states()
        train_mask_accuracy.reset_states()
        train_nsp_loss.reset_states()
        train_nsp_accuracy.reset_states()

        if cur_step % num_batches_checkpoint == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {}, batch {} at {}'.format(epoch, batch, ckpt_save_path))

        if cur_step > num_steps:
            break

    print ('Time taken training: {} secs\n'.format(time.time() - start))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    signal.signal(signal.SIGTERM, signal_handler)

    main()