import tensorflow as tf
import time
import click
from tensorflow_addons.optimizers import LAMB
import numpy as np

from dataset.ASA import create_dataset
from model import BERTTransformer, ASAClassifier

#loss_object = tf.keras.losses.MeanSquaredError(reduction='none')
loss_object = tf.keras.losses.MeanAbsoluteError(reduction='none')

def loss_function(real, pred, loss_weights):
    # tensorflow losses, even when given reduction none, will eat the last dimension
    # in this case we want losses per symbol, but real and pred have shape [num_batch, num_symbols]
    # meaning the loss would return [num_batch]
    # we solve this by adding a singleton dimension at the end, so pred and real have shape [num_batch, num_symbols, 1]
    # and the loss will reduce this to [num_batch, num_symbols]
    real = tf.expand_dims(real, 2)
    pred = tf.expand_dims(pred, 2)

    loss_ = loss_object(real, pred)
    loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)

    loss_ *= loss_weights

    return tf.reduce_sum(loss_)/tf.reduce_sum(loss_weights)

def loss_function2(real, pred):
    # tensorflow losses, even when given reduction none, will eat the last dimension
    # in this case we want losses per symbol, but real and pred have shape [num_batch, num_symbols]
    # meaning the loss would return [num_batch]
    # we solve this by adding a singleton dimension at the end, so pred and real have shape [num_batch, num_symbols, 1]
    # and the loss will reduce this to [num_batch, num_symbols]

    #Not need to do >4 but need to do >0 because only first one is special token now.
    #change if we do /100

    #sample_weight = tf.cast(tf.math.greater(real, 0.0), dtype=real.dtype)
    sample_weight = tf.cast(tf.math.greater(real, 0), dtype=real.dtype)
    real_ = tf.reshape(real, [tf.shape(real)[0]*tf.shape(real)[1]])
    pred_ = tf.reshape(pred, [tf.shape(pred)[0]*tf.shape(pred)[1]])

    sample_weight_ = tf.reshape(sample_weight, [tf.shape(sample_weight)[0]*tf.shape(sample_weight)[1]])
    indices = tf.squeeze(tf.where(tf.math.not_equal(sample_weight_, 0)), 1)
    targets = tf.gather(real_, indices)
    predictions = tf.gather(pred_, indices)

    loss_ = loss_object(targets, predictions)

    return loss_

train_mae = tf.keras.metrics.MeanAbsoluteError(name = 'train_mae')
val_mae = tf.keras.metrics.MeanAbsoluteError(name = 'val_mae')

# train_mse = tf.keras.metrics.MeanSquaredError(name = 'train_mse')
# val_mse = tf.keras.metrics.MeanSquaredError(name = 'val_mse')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
# ]


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
        writer = None

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        pad_mask = tf.math.logical_not(tf.math.equal(inp, 0))
        loss_weights = tf.cast(pad_mask, tf.float32)

        with tf.GradientTape() as tape:
            _, activations = transformer(inp, True, True)
            activations = activations[:, :, -1, :]
            predictions = classifier(inp, activations, True)

            loss_1 = loss_function(tar, predictions, loss_weights)
            loss = loss_function2(tar, predictions)
            if mixed_float:
                loss = optimizer.get_scaled_loss(loss)

        trainable_variables = list(classifier.trainable_variables)
        if not freeze_pretrained:
            trainable_variables.extend(transformer.trainable_variables)

        gradients = tape.gradient(loss, trainable_variables)
        if mixed_float:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss

    @tf.function(input_signature=train_step_signature)
    def performance_step(inp, tar):
        _, activations = transformer(inp, False, True)
        activations = activations[:, :, -1, :]
        predictions = classifier(inp, activations, False)

        targets_ = tf.reshape(tar, [tf.shape(tar)[0]*tf.shape(tar)[1]])
        predictions_ = tf.reshape(predictions, [tf.shape(predictions)[0]*tf.shape(predictions)[1]])

        #FIRST WAY
        loss_weights = tf.cast(tf.math.logical_not(tf.math.equal(inp, 0)), tf.float32)
        loss_weights_ = tf.reshape(loss_weights, [tf.shape(loss_weights)[0]*tf.shape(loss_weights)[1]])
        indices = tf.squeeze(tf.where(tf.math.not_equal(loss_weights_, 0)), 1)
        tar_ = tf.gather(targets_, indices)
        pred_ = tf.gather(predictions_, indices)

        #SECOND WAY
        #Not need to do >4 but need to do >0 because only first one is special token now.
        #change if we do /100
        #sample_weight=tf.cast(tf.math.greater(tar, 0.0), dtype=tar.dtype)
        sample_weight=tf.cast(tf.math.greater(tar, 0), dtype=tar.dtype)
        sample_weight_ = tf.reshape(sample_weight, [tf.shape(sample_weight)[0]*tf.shape(sample_weight)[1]])
        indices2 = tf.squeeze(tf.where(tf.math.not_equal(sample_weight_, 0)), 1)
        tar_2 = tf.gather(targets_, indices2)
        pred_2 = tf.gather(predictions_, indices2)

        return tar_2, pred_2


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
    classifier = ASAClassifier(dff=d_ff, rate=dropout_rate)

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

################################################################################
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

                #accuracy_step(accuracy_inp, accuracy_tar)
                #performance_step(accuracy_inp, accuracy_tar)
                targets, predictions = performance_step(accuracy_inp, accuracy_tar)
                train_mae(targets, predictions)

                #sample_weight=tf.cast(tf.math.greater(targets, 4), dtype=targets.dtype)
                #train_mae(targets, predictions, sample_weight=sample_weight)


                print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} MAE {:.4f}'.format(
                    cur_step, epoch + 1, batch, seqs_per_sec, train_mae.result()))

                if writer:
                    with writer.as_default():
                        tf.summary.scalar('Mean Absolute Error', train_mae.result(), cur_step)

                train_mae.reset_states()
            else:
                loss = train_step(inp, tar)

                if writer:
                    with writer.as_default():
                        tf.summary.scalar('Loss (MAE)', loss, cur_step)

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
        targets_val, predictions_val = performance_step(inp_val, tar_val)
        val_mae.update_state(targets_val, predictions_val)

    print("Performance over validation: MAE {:.4f}".format(val_mae.result()))

    print ('Time taken training: {} secs\n'.format(time.time() - start))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
