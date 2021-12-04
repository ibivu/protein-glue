import os

# Reset the TF_CONFIG environment variable
os.environ.pop('TF_CONFIG', None)

import tensorflow as tf
import time
import click
import json
import signal
import random
import hostlist
from datetime import datetime
from tensorflow_addons.optimizers import LAMB

from dataset.seqs import create_dataset, dataset_iter, dataset_iter_train_subset
from model.bert import BERTTransformer, FinalLayer, create_prediction_mask


############################################################################################################
# Parallelization Set-up

node_list = os.getenv("SLURM_JOB_NODELIST")
current_node = os.getenv("SLURMD_NODENAME")
num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
port_number = 3333

# Extracting the information from nodeList and formatting it for TF_CONFIG
node_list_correct = hostlist.expand_hostlist(node_list)
node_list_ports = []

for node in node_list_correct:
    # Add port_number to strings
    node_port = node + ":" + str(port_number)
    node_list_ports.append(node_port)

current_node_index = node_list_correct.index(current_node)

# Setting up dictionary for cluster
cluster_dict = {
    'worker': node_list_ports
}
task_index = current_node_index

# Setting environment variable TF_CONFIG
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': cluster_dict,
    'task': {'type': 'worker', 'index': task_index}
})

tf_config = json.loads(os.environ['TF_CONFIG'])
print(tf_config)

num_workers = len(tf_config['cluster']['worker'])

# Distribution strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def _is_chief(task_type, task_id):
    # If `task_type` is None, this may be operating as single worker, which works
    # effectively as chief.
    return task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0)

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id):
    if not _is_chief(task_type, task_id):
        filepath = _get_temp_dir(filepath, task_id)
    return filepath

is_chief = _is_chief(strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id) # True for chief-node


############################################################################################################
# Initial set-up

kill_signal = False

def signal_handler(sig, frame):
    print('Stop me!')

    global kill_signal
    kill_signal = True

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int64),
]

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

POSITIONAL_ENCODING_MAX_LENGTH = 520
LOGGING_EVERY_STEPS = 1000
NUM_SPECIAL_TOKENS = 20

@click.command()
@click.argument('input_files', nargs=-1)
@click.argument('output_dir', nargs=1)
@click.option('--val-files', default=None)
@click.option('--big-val-file', default=None)
@click.option('--learning-rate', default=1e-4)
@click.option('--num-layers', default=12)
@click.option('--num-heads', default=8)
@click.option('--d-ff', default=-1)
@click.option('--d-model', default=768)
@click.option('--dropout-rate', default=0.1)
@click.option('--batch-size', default=32)
@click.option('--keep-checkpoints', default=5)
@click.option('--num-batches-checkpoint', default=5000)
@click.option('--num-epochs', default=5)
@click.option('--num-steps', default=1000000)
@click.option('--big-input-file', default=None)
@click.option('--big-batch-size', default=4)
@click.option('--predict-rate', default=0.15)
@click.option('--predict-random-rate', default=0.1)
@click.option('--predict-nop-rate', default=0.1)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
@click.option('--sample-weights/--no-sample-weights', default=True)
@click.option('--mixed-float/--no-mixed-float', default=False)
@click.option('--tensorboard-dir', default=None)
def main(learning_rate, num_layers, num_heads, d_ff, d_model, dropout_rate, batch_size, keep_checkpoints,
         num_batches_checkpoint, num_epochs, num_steps, predict_rate, predict_random_rate, predict_nop_rate,
         reduced_target_alphabet, sample_weights, mixed_float, big_input_file, big_batch_size, input_files, output_dir, val_files, big_val_file,
         tensorboard_dir):

    if mixed_float:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if d_ff == -1:
        d_ff = d_model * 4

    input_vocab_size = 26 + NUM_SPECIAL_TOKENS
    target_vocab_size = 8 + NUM_SPECIAL_TOKENS if reduced_target_alphabet else 26 + NUM_SPECIAL_TOKENS

    token_counts = tf.Variable(tf.zeros((target_vocab_size,), dtype=tf.int64), trainable=False)


    ############################################################################################################
    # Datasets and distribution

    global_batch_size = batch_size * strategy.num_replicas_in_sync
    global_big_batch_size = big_batch_size * strategy.num_replicas_in_sync

    ds = create_dataset(
        input_files,
        batch_size=global_batch_size,
        target_reduced_alphabet=reduced_target_alphabet,
        max_sequence_length=128,
    )
    ds_big = create_dataset(
        [big_input_file],
        batch_size=global_big_batch_size,
        target_reduced_alphabet=reduced_target_alphabet,
        max_sequence_length=512,
    ) if big_input_file else None

    val_ds = create_dataset(
        val_files,
        batch_size=global_batch_size,
        target_reduced_alphabet=reduced_target_alphabet,
        max_sequence_length=128,
    )
    val_ds_big = create_dataset(
        [big_val_file],
        batch_size=global_big_batch_size,
        target_reduced_alphabet=reduced_target_alphabet,
        max_sequence_length=512,
    ) if big_val_file else None

    # Setting auto-sharding to DATA so that it shards by elements produced by the dataset (.FILE resulted in error)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    ds = ds.with_options(options)
    if big_input_file:
        ds_big = ds_big.with_options(options)

    val_ds = val_ds.with_options(options)
    if big_val_file:
        val_ds_big = val_ds_big.with_options(options)

    dataset_dist = strategy.experimental_distribute_dataset(ds)
    if big_input_file:
        dataset_big_dist = strategy.experimental_distribute_dataset(ds_big)

    val_dataset_dist = strategy.experimental_distribute_dataset(val_ds)
    if big_val_file:
        val_dataset_big_dist = strategy.experimental_distribute_dataset(val_ds_big)

    print("Counting number of batches:")
    num_batches = 0
    if big_input_file:
        num_batches_big = 0
    val_num_batches = 0
    if big_val_file:
        val_num_batches_big = 0

    for element in dataset_dist:
        num_batches += 1
    if big_input_file:
        for element in dataset_big_dist:
            num_batches_big += 1
    for batch in val_dataset_dist:
        val_num_batches += 1
    if big_val_file:
        for batch in val_dataset_big_dist:
            val_num_batches_big += 1

    print("Number of batches in dataset: %s" % num_batches)
    if big_input_file:
        print("Number of batches in dataset_big: %s" % num_batches_big)
    print("Number of batches in val_dataset: %s" % val_num_batches)
    if big_val_file:
        print("Number of batches in val_dataset_big: %s" % val_num_batches_big)

    # Set up subset of trainingset for validation process
    temp_batches_training = [s for s in range(1,num_batches) if s%10 !=5]
    selected_batches_training = random.sample(temp_batches_training, val_num_batches)
    temp_batches_training_big = [s for s in range(1,max(selected_batches_training)) if s%10 ==5]
    selected_batches_training_big = random.sample(temp_batches_training_big, int(len(selected_batches_training)/10)+1)

    selected_batches_training.sort()
    selected_batches_training_big.sort()


    ############################################################################################################
    # Loss, optimizer, and model set-up

    # Used by tensorboard to write loss values to
    if tensorboard_dir:
        print("Creating Tensorboard")
        tensorboard_dir_worker = tensorboard_dir + '/' + str(strategy.cluster_resolver.task_id)
        writer = tf.summary.create_file_writer(tensorboard_dir_worker)

        if is_chief:
            tensorboard_dir_val_worker = tensorboard_dir_worker + '_val'
            tensorboard_dir_subtrain_worker = tensorboard_dir_worker + '_subtrain'
            val_writer = tf.summary.create_file_writer(tensorboard_dir_val_worker)
            subtrain_writer = tf.summary.create_file_writer(tensorboard_dir_subtrain_worker)
    else:
        writer = None

    # Variables and model created in strategy.scope() are replicated on each worker
    with strategy.scope():

        # Create the loss and other metrics
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # Per-symbol loss
        def masked_loss_function(real, pred, loss_weights):
            loss_ = loss_object(real, pred)
            loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)
            loss_ *= loss_weights

            return tf.reduce_sum(loss_) / tf.reduce_sum(loss_weights) / strategy.num_replicas_in_sync

        def nsp_loss_function(real, pred):
            # Slice out first token prediction
            sliced_real = real[:, 0]  # (batch_size, 1)
            sliced_pred = pred[:, 0, :]  # (batch_size, 1, 2)

            loss_ = loss_object(sliced_real, sliced_pred)

            current_batch_size = tf.shape(real)[0]
            current_batch_size = tf.cast(current_batch_size, dtype=tf.float32)

            return tf.reduce_sum(loss_) / current_batch_size / strategy.num_replicas_in_sync

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        nsp_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='nsp_train_accuracy')

        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        val_nsp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='nsp_val_accuracy')

        # Create the model
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
        masked_token_predictor = FinalLayer(target_vocab_size)
        is_next_predictor = FinalLayer(2)  # 0 or 1

        # Create the optimizer
        # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(learning_rate, num_steps, end_learning_rate=0.0, power=1.0)
        optimizer = LAMB(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay_rate=0.01)

        if mixed_float:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')

        # Set-up Checkpointing
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    epoch_ckpt_dir = output_dir + 'epochs'

    write_ckpt_dir = write_filepath(output_dir, strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
    write_epoch_ckpt_dir = write_filepath(epoch_ckpt_dir, strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)

    ckpt_manager = tf.train.CheckpointManager(ckpt, write_ckpt_dir, max_to_keep=keep_checkpoints)  # Save regular checkpoints
    epoch_ckpt_manager = tf.train.CheckpointManager(ckpt, write_epoch_ckpt_dir, max_to_keep=keep_checkpoints)  # Save checkpoints after every epoch


    ############################################################################################################
    # Training function

    def train_step(inp, tar, is_next, segment_label):
        nonlocal token_counts

        inp, _, pad_prediction_mask = create_prediction_mask(
            inp,
            input_vocab_size=input_vocab_size,
            predict_rate=predict_rate,
            predict_nop_rate=predict_nop_rate,
            predict_random_rate=predict_random_rate
        )

        # Update the token counts with the real tokens we've been asked to predict
        predict_tokens = tf.gather_nd(tar, tf.where(pad_prediction_mask))
        new_token_idxs, _, new_token_counts = tf.unique_with_counts(predict_tokens, out_idx=tf.int64)
        token_counts.assign_add(tf.scatter_nd(tf.expand_dims(new_token_idxs, -1), new_token_counts, shape=(target_vocab_size,)))

        # Use the current token counts to scale the loss, so the more common amino acid types are weighted approximately
        # equally to the less occurring ones. Add a constant to the token fractions so we never divide by zero.
        token_fracs = tf.cast(token_counts / tf.reduce_sum(token_counts), tf.float32)
        random_frac = 1 / (target_vocab_size - NUM_SPECIAL_TOKENS)
        token_propensities =  random_frac / token_fracs
        inf_idxs = tf.where(tf.logical_not(tf.math.is_finite(token_propensities)))
        token_propensities = tf.tensor_scatter_nd_update(token_propensities, inf_idxs, tf.fill((tf.shape(inf_idxs)[0],), 1.0))
        if sample_weights:
            loss_weights = tf.gather(token_propensities, tar) * tf.cast(pad_prediction_mask, tf.float32)
        else:
            loss_weights = tf.cast(pad_prediction_mask, tf.float32)

        # Loss per symbol used for updating gradients
        with tf.GradientTape() as tape:
            embeddings = transformer(inp, segment_label, True)
            token_predictions = masked_token_predictor(embeddings)
            masked_token_loss = masked_loss_function(tar, token_predictions, loss_weights)
            is_next_prediction = is_next_predictor(embeddings)
            nsp_loss = nsp_loss_function(is_next, is_next_prediction)
            loss = masked_token_loss + nsp_loss  # Add masked token loss and NSP loss
            if mixed_float:
                loss = optimizer.get_scaled_loss(loss)

        trainable_variables = list(masked_token_predictor.trainable_variables)
        trainable_variables.extend(is_next_predictor.trainable_variables)
        trainable_variables.extend(transformer.trainable_variables)

        gradients = tape.gradient(loss, trainable_variables)
        if mixed_float:
            gradients = optimizer.get_unscaled_gradients(gradients)

        # Gradient clipping
        # gradients = [tf.clip_by_norm(g, 1) for g in gradients]

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Gradient norm
        gradient_norm = tf.linalg.global_norm(gradients)

        # Don't include the padded or non-prediction positions in the accuracy.
        mask = tf.cast(pad_prediction_mask, dtype=tf.float32)

        train_accuracy.update_state(tar, token_predictions, sample_weight=mask)
        nsp_train_accuracy.update_state(is_next[:, 0], is_next_prediction[:, 0, :])

        return loss, masked_token_loss, nsp_loss, gradient_norm


    @tf.function(input_signature=train_step_signature)
    def distributed_train_step(inp, tar, is_next, segment_label, cur_step):

        per_replica_loss, per_replica_masked_token_loss, per_replica_nsp_loss, per_replica_gradient_norms = strategy.run(train_step, args=(inp, tar, is_next, segment_label, ))

        if writer:
            with writer.as_default():
                tf.summary.scalar('per_worker_loss', per_replica_loss, cur_step)
                tf.summary.scalar('per_worker_masked_token_loss', per_replica_masked_token_loss, cur_step)
                tf.summary.scalar('per_worker_nsp_loss', per_replica_nsp_loss, cur_step)
                #tf.summary.scalar('per_worker_gradient_norm', per_replica_gradient_norms, cur_step)

        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)  # per_replica_loss is summed across all workers

        return reduced_loss


    ############################################################################################################
    # Validation functions

    def val_step(inp, tar, is_next, segment_label):
        nonlocal token_counts

        inp, _, pad_prediction_mask = create_prediction_mask(
            inp,
            input_vocab_size=input_vocab_size,
            predict_rate=predict_rate,
            predict_nop_rate=predict_nop_rate,
            predict_random_rate=predict_random_rate
        )

        # Update the token counts with the real tokens we've been asked to predict
        predict_tokens = tf.gather_nd(tar, tf.where(pad_prediction_mask))
        new_token_idxs, _, new_token_counts = tf.unique_with_counts(predict_tokens, out_idx=tf.int64)
        token_counts.assign_add(tf.scatter_nd(tf.expand_dims(new_token_idxs, -1), new_token_counts, shape=(target_vocab_size,)))

        # Use the current token counts to scale the loss, so the more common amino acid types are weighted approximately
        # equally to the less occurring ones. Add a constant to the token fractions so we never divide by zero.
        token_fracs = tf.cast(token_counts / tf.reduce_sum(token_counts), tf.float32)
        random_frac = 1 / (target_vocab_size - NUM_SPECIAL_TOKENS)
        token_propensities =  random_frac / token_fracs
        inf_idxs = tf.where(tf.logical_not(tf.math.is_finite(token_propensities)))
        token_propensities = tf.tensor_scatter_nd_update(token_propensities, inf_idxs, tf.fill((tf.shape(inf_idxs)[0],), 1.0))
        if sample_weights:
            loss_weights = tf.gather(token_propensities, tar) * tf.cast(pad_prediction_mask, tf.float32)
        else:
            loss_weights = tf.cast(pad_prediction_mask, tf.float32)

        # Calculate loss without calculating gradients
        embeddings = transformer(inp, segment_label, True)
        token_predictions = masked_token_predictor(embeddings)
        masked_token_loss = masked_loss_function(tar, token_predictions, loss_weights)
        is_next_prediction = is_next_predictor(embeddings)
        nsp_loss = nsp_loss_function(is_next, is_next_prediction)
        loss = masked_token_loss + nsp_loss  # Add masked token loss and NSP loss
        if mixed_float:
            loss = optimizer.get_scaled_loss(loss)

        # Don't include the padded or non-prediction positions in the accuracy.
        mask = tf.cast(pad_prediction_mask, dtype=tf.float32)

        val_accuracy.update_state(tar, token_predictions, sample_weight=mask)
        val_nsp_accuracy.update_state(is_next[:, 0], is_next_prediction[:, 0, :])

        return loss, masked_token_loss, nsp_loss

    @tf.function(input_signature=val_step_signature)
    def distributed_val_step(val_inp, val_tar, val_is_next, val_segment_label):
        per_replica_val_loss, per_replica_val_masked_token_loss, per_replica_val_nsp_loss = strategy.run(val_step, args=(val_inp, val_tar, val_is_next, val_segment_label, ))

        # if val_writer:
        #     with val_writer.as_default():
        #         tf.summary.scalar('per_worker_val_loss', per_replica_val_loss, cur_step)

        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_loss, axis=None)


    ############################################################################################################
    # Training

    # if a checkpoint exists, restore the latest checkpoint.
    print('Looking for checkpoints')
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    start = time.time()
    start_batch = time.time()
    seqs_batch = 0

    cur_epoch = 1

    # For epoch average
    num_batches_counter = 0
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    total_train_nsp_accuracy = 0.0

    print('Starting training...')

    for epoch, batch, inp, tar, is_next, segment_label in dataset_iter(dataset_dist, dataset_big_dist):
        if epoch > num_epochs:
            break

        global kill_signal
        if kill_signal == False:

            # cur_epoch gets updated after validation so while we're in the same epoch do training
            if cur_epoch == epoch:

                cur_step = optimizer.iterations.numpy() + 1
                seqs_batch += inp.shape[0]

                # is_large = False
                # if inp.shape[1] > 128:
                #     is_large = True
                # else:
                #     is_large = False

                train_loss = distributed_train_step(inp, tar, is_next, segment_label, cur_step)

                # For tensorboard logging
                masked_accuracy = train_accuracy.result()
                nsp_accuracy = nsp_train_accuracy.result()

                # For epoch average
                num_batches_counter += 1
                total_train_loss += train_loss
                total_train_accuracy += masked_accuracy
                total_train_nsp_accuracy += nsp_accuracy

                if is_chief:
                    if writer:
                        with writer.as_default():
                            tf.summary.scalar('learning_rate', optimizer._decayed_lr('float32').numpy(), cur_step)
                            tf.summary.scalar('masked_accuracy', masked_accuracy, cur_step) # Captures accuracy as y-value and cur_step as x-value for tensorboard
                            tf.summary.scalar('nsp_accuracy', nsp_accuracy, cur_step)
                            # tf.summary.scalar('train_loss', train_loss, cur_step)
                            # if is_large:
                            #     tf.summary.scalar('binary_plot_big', 1, cur_step)
                            # else:
                            #     tf.summary.scalar('binary_plot_big', 0, cur_step)

                # if is_chief:
                if cur_step % LOGGING_EVERY_STEPS == 0:
                    now = time.time()
                    seqs_per_sec = seqs_batch / (now - start_batch)
                    start_batch = now
                    seqs_batch = 0

                    print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.4f} Symbol_Loss {:.4f} Accuracy {:.4f} NSP_Accuracy: {:.4f}'.format(
                        cur_step, epoch, batch, seqs_per_sec, train_loss, masked_accuracy, nsp_accuracy))

                # Reset metrics after every batch
                train_accuracy.reset_states()
                nsp_train_accuracy.reset_states()

                # Save checkpoint
                if cur_step % num_batches_checkpoint == 0:
                    ckpt_save_path = ckpt_manager.save()
                    if not is_chief:
                        tf.io.gfile.rmtree(write_ckpt_dir)
                    print ('Saved checkpoint for epoch {}, batch {} at {}'.format(epoch, batch, ckpt_save_path))

                # For DAS, check every 200 batches if it's time to stop training
                if cur_step % 200 == 0:
                    now_hour = int(str(datetime.now().time())[0:2]) # time object
                    weekday = datetime.today().weekday()
                    if weekday != 5 and weekday != 6 and now_hour == 8:
                        print("Time is up!")
                        kill_signal = True


            ############################################################################################################
            # End of epoch summary

            # At the end of epoch do validation (for testing also after certain number of batches)
            if (cur_epoch != epoch) or (batch % 10000 == 0):
                average_epoch_train_loss = total_train_loss / num_batches_counter
                average_epoch_train_accuracy = total_train_accuracy / num_batches_counter
                average_epoch_train_nsp_accuracy = total_train_nsp_accuracy / num_batches_counter

                # Reset variables
                num_batches_counter = 0
                total_train_loss = 0
                total_train_accuracy = 0
                total_train_nsp_accuracy = 0

                if is_chief:
                    if writer:
                        with writer.as_default():
                            tf.summary.scalar('epoch_loss', average_epoch_train_loss, cur_step)
                            tf.summary.scalar('epoch_masked_accuracy', average_epoch_train_accuracy, cur_step)
                            tf.summary.scalar('epoch_nsp_accuracy', average_epoch_train_nsp_accuracy, cur_step)

                print ('Epoch {} Average Train_Loss {:.4f} Average Train_Masked_Accuracy {:.4f} Average Train_NSP_Accuracy {:.4f}'.format(
                    cur_epoch, average_epoch_train_loss, average_epoch_train_accuracy, average_epoch_train_nsp_accuracy))


                ############################################################################################################
                # Testing the training subset on last gradients

                print('Starting training-subset validation...')

                num_subtrain_batches_counter = 0
                total_subtrain_loss = 0
                total_subtrain_accuracy = 0
                total_subtrain_nsp_accuracy = 0

                # Subtrain loop
                for subtrain_epoch, subtrain_batch, subtrain_inp, subtrain_tar, subtrain_is_next, subtrain_segment_label in dataset_iter_train_subset(dataset_dist, dataset_big_dist, selected_batches_training, selected_batches_training_big):
                    if subtrain_epoch == 2:
                        print('Ending training-subset validation...')
                        break

                    subtrain_loss = distributed_val_step(subtrain_inp, subtrain_tar, subtrain_is_next, subtrain_segment_label)

                    num_subtrain_batches_counter += 1
                    total_subtrain_loss += subtrain_loss
                    total_subtrain_accuracy += val_accuracy.result()
                    total_subtrain_nsp_accuracy += val_nsp_accuracy.result()

                average_epoch_subtrain_loss = total_subtrain_loss / num_subtrain_batches_counter
                average_epoch_subtrain_accuracy = total_subtrain_accuracy / num_subtrain_batches_counter
                average_epoch_subtrain_nsp_accuracy = total_subtrain_nsp_accuracy / num_subtrain_batches_counter

                # if is_chief:
                print ('Epoch {} Subtrain_Loss {:.4f} Subtrain_Masked_Accuracy {:.4f} Subtrain_NSP_Accuracy: {:.4f}'.format(
                    epoch, average_epoch_subtrain_loss, average_epoch_subtrain_accuracy,
                    average_epoch_subtrain_nsp_accuracy))

                if is_chief:
                    if subtrain_writer:
                        with subtrain_writer.as_default():
                            tf.summary.scalar('epoch_loss', average_epoch_subtrain_loss, cur_step)
                            tf.summary.scalar('epoch_masked_accuracy', average_epoch_subtrain_accuracy, cur_step)
                            tf.summary.scalar('epoch_nsp_accuracy', average_epoch_subtrain_nsp_accuracy, cur_step)

                # Reset metrics after every batch
                val_accuracy.reset_states()
                val_nsp_accuracy.reset_states()


                ############################################################################################################
                # Validation

                print('Starting validation...')

                num_val_batches_counter = 0
                total_val_loss = 0
                total_val_accuracy = 0
                total_val_nsp_accuracy = 0

                # Validation loop
                for val_epoch, val_batch, val_inp, val_tar, val_is_next, val_segment_label in dataset_iter(val_dataset_dist, val_dataset_big_dist):
                    if val_epoch == 2:
                        print('Ending validation...')
                        break

                    val_loss = distributed_val_step(val_inp, val_tar, val_is_next, val_segment_label)

                    num_val_batches_counter += 1
                    total_val_loss += val_loss
                    total_val_accuracy += val_accuracy.result()
                    total_val_nsp_accuracy += val_nsp_accuracy.result()

                average_epoch_val_loss = total_val_loss / num_val_batches_counter
                average_epoch_val_accuracy = total_val_accuracy / num_val_batches_counter
                average_epoch_val_nsp_accuracy = total_val_nsp_accuracy / num_val_batches_counter

                # if is_chief:
                print ('Epoch {} Val_Loss {:.4f} Val_Accuracy {:.4f} Val_NSP_Accuracy: {:.4f}'.format(
                    cur_epoch, average_epoch_val_loss, average_epoch_val_accuracy,
                    average_epoch_val_nsp_accuracy))

                if is_chief:
                    if val_writer:
                        with val_writer.as_default():
                            tf.summary.scalar('epoch_loss', average_epoch_val_loss, cur_step)
                            tf.summary.scalar('epoch_masked_accuracy', average_epoch_val_accuracy, cur_step)
                            tf.summary.scalar('epoch_nsp_accuracy', average_epoch_val_nsp_accuracy, cur_step)

                # Reset metrics after every batch
                val_accuracy.reset_states()
                val_nsp_accuracy.reset_states()

                # Save epoch-checkpoint
                epoch_ckpt_save_path = epoch_ckpt_manager.save()
                if not is_chief:
                    tf.io.gfile.rmtree(write_epoch_ckpt_dir)
                print ('Saved checkpoint for epoch {}, batch {} at {}'.format(epoch, batch, epoch_ckpt_save_path))

            # Update cur_epoch to epoch and go back to training
            cur_epoch = epoch

        # If kill_signal == True
        else:
            print('Job was killed! Stop training...')
            ckpt_save_path = ckpt_manager.save()
            if not is_chief:
                tf.io.gfile.rmtree(write_ckpt_dir)
            print ('Saved checkpoint for epoch {}, batch {} at {}'.format(epoch, batch, ckpt_save_path))

            break

        if cur_step > num_steps:
            print('Reached total number of steps! Stop training...')
            ckpt_save_path = ckpt_manager.save()
            if not is_chief:
                tf.io.gfile.rmtree(write_ckpt_dir)
            print ('Saved checkpoint for epoch {}, batch {} at {}'.format(epoch, batch, ckpt_save_path))

            break

    print ('Time taken training: {} secs\n'.format(time.time() - start))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter

    signal.signal(signal.SIGTERM, signal_handler)

    main()
