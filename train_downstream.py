## import
import tensorflow as tf
import numpy as np
import time
import click
from tensorflow_addons.optimizers import LAMB

from dataset import create_dataset_ss3, create_dataset_ss8, create_dataset_asa, create_dataset_bur, create_dataset_ppi, create_dataset_epi, create_dataset_pni, create_dataset_psmi, create_dataset_hpc, create_dataset_hpcr, create_dataset_hpr
from model import BERTTransformer, NSPEmbeddingLayer, SS3Classifier, SS8Classifier, BURClassifier, ASAClassifier, PPIClassifier, EPIClassifier, PNIClassifier, PSMIClassifier, HPCClassifier, HPRClassifier, HPCRClassifier
from utils_downstream import performance_step_regression, performance_step_interface, correct_formatting_interface, metrices_interface, plotting_pr, plotting_roc, write_output_plots
import constants as c
import constants_downstream as cd


## specify which task
@click.command()
@click.argument('input_files', nargs=-1)
@click.argument('output_dir', nargs=1)
@click.option('--task', default='ss3')
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
@click.option('--num-batches-checkpoint', default=10000)
@click.option('--num-epochs', default=5)
@click.option('--num-steps', default=1000000)
@click.option('--max_length_seq', default=512)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
@click.option('--mixed-float/--no-mixed-float', default=False)
@click.option('--freeze-pretrained/--no-freeze-pretrained', default=False)
@click.option('--tensorboard-dir', default=None)
@click.option('--summary-dir', default=None)

def main(learning_rate, num_layers, num_heads, d_ff, d_model, dropout_rate, batch_size, keep_checkpoints, pretrain_checkpoint_dir,
         num_batches_checkpoint, num_epochs, num_steps, mixed_float, reduced_target_alphabet, validation_file, input_files, output_dir, task, max_length_seq,
         freeze_pretrained, tensorboard_dir, summary_dir):

    print("Downstream task: {}".format(task))
    print("learning rate: {}".format(learning_rate))
    print("batch size: {}".format(batch_size))
    print("dropout rate: {}".format(dropout_rate))

    #####
    if task not in ["ss3", "ss8", "bur", "ppi", "epi", "pni", "psmi", "hpc", "hpcr","asa", "hpr"]:
        print("Not a possible classification task")

    classification_task = ["ss3", "ss8", "bur", "ppi", "epi", "pni", "psmi", "hpc", "hpcr"]
    regression_task = ["asa", "hpr"]
    interface_task = [ "ppi", "epi", "pni", "psmi"]

    """
    Classification: SparseCategoricalCrossentropy
    Regression: Mean Absolute Error
    """

    if task in regression_task:
        loss_object = tf.keras.losses.MeanAbsoluteError(reduction='none')
        train_mae = tf.keras.metrics.MeanAbsoluteError(name = 'train_mae')
        val_mae = tf.keras.metrics.MeanAbsoluteError(name = 'val_mae')
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    if task in interface_task:
        train_FP = tf.keras.metrics.FalsePositives(name='train_FP')
        train_TP = tf.keras.metrics.TruePositives(name = 'train_TP')
        train_FN = tf.keras.metrics.FalseNegatives(name = 'train_FN')
        train_TN = tf.keras.metrics.TrueNegatives(name = 'train_TN')

        val_FP = tf.keras.metrics.FalsePositives(name='val_FP')
        val_TP = tf.keras.metrics.TruePositives(name = 'val_TP')
        val_FN = tf.keras.metrics.FalseNegatives(name = 'val_FN')
        val_TN = tf.keras.metrics.TrueNegatives(name = 'val_TN')

    def loss_function_classification(real, pred, loss_weights):
        loss_ = loss_object(real, pred)
        loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)
        loss_ *= loss_weights

        return tf.reduce_sum(loss_)/tf.reduce_sum(loss_weights)

    def loss_function_regression(real, pred, class_imbalance):
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

        if class_imbalance:
            majority_class_weight = class_imbalance[0]
            minority_class_weight = class_imbalance[1]

            loss_weights = tf.zeros([tf.shape(targets)[0]], tf.float32)
            loss_weights = tf.where(tf.math.not_equal(targets, 1), loss_weights, [majority_class_weight])
            loss_weights = tf.where(tf.math.less_equal(targets, 1), loss_weights, [minority_class_weight])
            targets_ =  tf.reshape(targets, [tf.shape(targets)[0],1])
            predictions_ =  tf.reshape(predictions, [tf.shape(predictions)[0],1])

            loss_ = loss_object(targets_, predictions_, loss_weights)
            ##loss_ = list. each element is abs(target_[i]-prediction_[i])*loss_weights
            #in order to get total loss: sum and divide by number of elements
            loss_ = tf.reduce_sum(loss_)/tf.dtypes.cast(tf.shape(loss_)[0], tf.float32)
        else:
            loss_ = loss_object(targets, predictions)

        return loss_


    if task == "asa":
        train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32)]
    else:
        train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32)]

    #####

    if mixed_float:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if d_ff == -1:
        d_ff = d_model * c.D_FF_D_MODEL_RATIO
    input_vocab_size = 26 + c.NUM_SPECIAL_SYMBOLS
    target_vocab_size = 8 + c.NUM_SPECIAL_SYMBOLS if reduced_target_alphabet else input_vocab_size

    if tensorboard_dir:
        print("Creating Tensorboard")
        print(tensorboard_dir)
        writer = tf.summary.create_file_writer(tensorboard_dir)
    else:
        writer = None

    #input_signature=train_step_signature_ci
    @tf.function(experimental_relax_shapes=True)
    def train_step(inp, tar, segment_label, class_imbalance, classification=True):
        pad_mask = tf.math.logical_not(tf.math.equal(inp, 0))
        loss_weights = tf.cast(pad_mask, tf.float32)

        if class_imbalance is not None and classification == True:          #note regression class imbalance in loss function regression
            majority_class_weight = class_imbalance[0]
            minority_class_weight = class_imbalance[1]
            loss_weights = tf.where(tf.math.not_equal(tar, 1), loss_weights, [majority_class_weight])
            loss_weights = tf.where(tf.math.not_equal(tar, 2), loss_weights, [minority_class_weight])

        with tf.GradientTape() as tape:
            inp_extra = nsp_embedding(segment_label)
            _, activations = transformer(inp, inp_extra, True, True)
            activations = activations[:, :, -1, :]
            predictions = classifier(inp, activations, True)

            if classification:
                loss = loss_function_classification(tar, predictions, loss_weights)
            else:
                loss = loss_function_regression(tar, predictions, class_imbalance)
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
    def accuracy_step(inp, tar, segment_label):
        inp_extra = nsp_embedding(segment_label)
        _, activations = transformer(inp, inp_extra, False, True)
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

    LOGGING_EVERY_STEPS = 10
    if task == "ss3":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_SS3
        classifier = SS3Classifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_ss3(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_ss3([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = None
    elif task == "ss8":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_SS3
        classifier = SS8Classifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_ss8(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_ss8([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = None
    elif task == "bur":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_BUR
        classifier = BURClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_bur(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_bur([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = None
    elif task == "asa":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_ASA
        classifier = ASAClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_asa(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_asa([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = None
    elif task == "ppi":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_PPI
        classifier = PPIClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_ppi(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_ppi([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = [cd.MAJORITY_CLASS_PPI, cd.MINORITY_CLASS_PPI]
    elif task == "epi":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_EPI
        classifier = EPIClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_epi(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_epi([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = [cd.MAJORITY_CLASS_EPI, cd.MINORITY_CLASS_EPI]
    elif task == "pni":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_PNI
        classifier = PNIClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_pni(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_pni([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = [cd.MAJORITY_CLASS_PNI, cd.MINORITY_CLASS_PNI]

        LOGGING_EVERY_STEPS = 5
    elif task == "psmi":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_PSMI
        classifier = PSMIClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_psmi(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_psmi([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = [cd.MAJORITY_CLASS_PSMI, cd.MINORITY_CLASS_PNI]
    elif task == "hpr":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_HPR
        classifier = HPRClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_hpr(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_hpr([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = [cd.MAJORITY_CLASS_HPR, cd.MINORITY_CLASS_HPR]
    elif task == "hpc":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_HPC
        classifier = HPCClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_hpc(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_hpc([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = None
    elif task == "hpcr":
        POSITIONAL_ENCODING_MAX_LENGTH = cd.POSITIONAL_ENCODING_MAX_LENGTH_HPCR
        classifier = HPCRClassifier(dff=d_ff, rate=dropout_rate)
        ds = create_dataset_hpcr(input_files, batch_size=batch_size, max_length=max_length_seq)
        validation_ds = create_dataset_hpcr([validation_file], batch_size=batch_size, max_length=max_length_seq) if validation_file else None

        class_imbalance = None

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

    if pretrain_checkpoint_dir:
        pretrain_ckpt = tf.train.Checkpoint(transformer=transformer)
        pretrain_ckpt.restore(tf.train.latest_checkpoint(pretrain_checkpoint_dir)).expect_partial()
        print("Loaded pre-trained model from checkpoint!")
        print(pretrain_checkpoint_dir)
    else:
        print("No pre-training loaded")

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(learning_rate, num_steps, end_learning_rate=0.0, power=1.0)
    optimizer = LAMB(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay_rate=0.01)
    if mixed_float:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    ckpt = tf.train.Checkpoint(transformer=transformer,
                            classifier=classifier,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=keep_checkpoints)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!')
        print(pretrain_checkpoint_dir)

################################################################################
    cur_step = None
    start = time.time()
    for epoch in range(num_epochs):
        batch = 1
        start_batch = time.time()
        validation_iter = iter(validation_ds) if validation_ds else None

        for (inp, tar, segment_label) in ds:
            cur_step = optimizer.iterations.numpy() + 1

            if batch % LOGGING_EVERY_STEPS == 0:
                now = time.time()
                seqs_per_sec = (batch_size * LOGGING_EVERY_STEPS) / (now - start_batch)
                start_batch = now

                try:
                    accuracy_inp, accuracy_tar, accuracy_segment_label = next(validation_iter) if validation_iter else (inp, tar, segment_label)
                except StopIteration:
                    validation_iter = iter(validation_ds) if validation_ds else None
                    accuracy_inp, accuracy_tar, accuracy_segment_label = next(validation_iter) if validation_iter else (inp, tar, segment_label)

                targets, predictions, weights = accuracy_step(accuracy_inp, accuracy_tar, accuracy_segment_label)

                if task in regression_task:
                    tar, pred = performance_step_regression(targets, predictions)
                    pcc = np.corrcoef(pred, tar)[0][1]
                    train_mae(tar, pred)

                    print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} MAE {:.4f} PCC {:.4f}'.format(
                        cur_step, epoch + 1, batch, seqs_per_sec, train_mae.result(), pcc))

                    if writer:
                        with writer.as_default():
                            tf.summary.scalar('Mean Absolute Error', train_mae.result(), cur_step)
                            tf.summary.scalar('Pearson Correlation Coefficient', pcc, cur_step)

                    train_mae.reset_states()

                elif task in interface_task:
                    tar_list, pred_max_1D, IF_pred_list = performance_step_interface(targets, predictions)
                    train_accuracy(targets, predictions, sample_weight = weights)
                    targets, predictions, pred_prob_IF = correct_formatting_interface(tar_list, pred_max_1D, IF_pred_list)
                    fpr, tpr, auc_roc, auc_pr, precision, recall, precision_list, recall_list, fraction_positive = metrices_interface(targets, predictions, pred_prob_IF)
                    train_TP(targets, predictions)
                    train_FP(targets, predictions)
                    train_FN (targets, predictions)
                    train_TN(targets, predictions)

                    print('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} Accuracy {:.2f} Precision {:.2f} Recall {:.2f} AUC_roc {:.2f} AUC_pr {:.2f} TP {:.1f} FP {:.1f} TN {:.1f} FN {:.1f}'.format(
                    cur_step, epoch + 1, batch, seqs_per_sec, train_accuracy.result(), precision, recall, auc_roc, auc_pr, train_TP.result(), train_FP.result(), train_TN.result(), train_FN .result()))

                    if writer:
                        with writer.as_default():
                            tf.summary.scalar('accuracy', train_accuracy.result(), cur_step)
                            tf.summary.scalar('Precision',precision, cur_step)
                            tf.summary.scalar('Recall', recall , cur_step)
                            tf.summary.scalar('AUC ROC', auc_roc, cur_step)
                            tf.summary.scalar('AUC PR', auc_pr, cur_step)

                    train_accuracy.reset_states()
                    train_TP.reset_states()
                    train_FP.reset_states()
                    train_TN.reset_states()
                    train_FN .reset_states()
                else:
                    train_accuracy(targets, predictions, sample_weight = weights)

                    print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} Accuracy {:.4f}'.format(
                        cur_step, epoch + 1, batch, seqs_per_sec, train_accuracy.result()))

                    if writer:
                        with writer.as_default():
                            tf.summary.scalar('accuracy', train_accuracy.result(), cur_step)

                    train_accuracy.reset_states()
            else:
                if task in classification_task:
                    loss = train_step(inp, tar, segment_label, class_imbalance, True)
                else:
                    loss = train_step(inp, tar, segment_label, class_imbalance, False)

                if writer:
                    with writer.as_default():
                        tf.summary.scalar('Loss', loss, cur_step)

            if cur_step % num_batches_checkpoint == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {}, batch {} at {}'.format(epoch + 1, batch, ckpt_save_path))

            batch += 1

            if cur_step > num_steps:
                break
        if cur_step > num_steps:
            break

################################################################################
    ## End validation
    if task in regression_task:
        targets_val_list = []
        predictions_val_list = []
        for (inp_val, tar_val, segment_label_val) in validation_ds:
            targets, predictions, weights = accuracy_step(inp_val, tar_val, segment_label_val)
            targets_val, predictions_val = performance_step_regression(targets, predictions)
            targets_val_list.extend(targets_val)
            predictions_val_list.extend(predictions_val)
            val_mae.update_state(targets_val, predictions_val)
        pcc_enval = np.corrcoef(predictions_val_list, targets_val_list)[0][1]
        print("Performance over validation: MAE {:.4f} PCC {:.4f}".format(val_mae.result(), pcc_enval))

        if summary_dir:
            summary_path = summary_dir + ".txt"
            summary_file = open(summary_path, "a")
            summary_file.write("{}: MAE: {}".format(task, val_mae.result()))
            summary_file.write("\n")
            summary_file.write("{}: PCC: {}".format(task, pcc_enval))
            summary_file.write("\n")
            summary_file.close()

    elif task in interface_task:
        tar_list_val = []
        pred_list_val = []
        prob_list_val = []

        for (inp_val, tar_val, segment_label_val) in validation_ds:
            targets, predictions, weights = accuracy_step(inp_val, tar_val, segment_label_val)
            val_accuracy(targets, predictions, sample_weight=weights)
            tar_list, pred_max_1D, IF_pred_list = performance_step_interface(targets, predictions)
            targets, predictions, pred_prob_IF = correct_formatting_interface(tar_list, pred_max_1D, IF_pred_list)
            tar_list_val.extend(targets)
            pred_list_val.extend(predictions)
            prob_list_val.extend(pred_prob_IF)

        #After all validation data has been seen.
        fpr_val, tpr_val, auc_roc_val, auc_pr_val, precision_val, recall_val, precision_list_val, recall_list_val, fraction_positive_val = metrices_interface(tar_list_val, pred_list_val, prob_list_val)
        val_FP(tar_list_val, pred_list_val)
        val_TP(tar_list_val, pred_list_val)
        val_FN(tar_list_val, pred_list_val)
        val_TN(tar_list_val, pred_list_val)

        print("creating plots")
        # path_fig_roc = tensorboard_dir + "/ROC.png"
        # path_fig_pr = tensorboard_dir + "/PR.png"
        path_output_write = tensorboard_dir + "/output.txt"

        # plotting_roc(fpr_val, tpr_val, auc_roc_val, path_fig_roc)
        # plotting_pr(recall_list_val, precision_list_val, fraction_positive_val, path_fig_pr)
        write_output_plots(auc_roc_val, auc_pr_val, fraction_positive_val, tar_list_val, pred_list_val, prob_list_val, path_output_write)

        print("Validation performance")
        print('Accuracy {:.4f} Precision {:.4f} Recall {:.4f} AUC_roc {:.4f} AUC_pr {:.4f} TP {:.1f} FP {:.1f} TN {:.1f} FN {:.1f}'.format(
        val_accuracy.result(), precision_val, recall_val, auc_roc_val, auc_pr_val, val_TP.result(), val_FP.result(), val_TN.result(), val_FN.result()))

        if summary_dir:
            summary_path = summary_dir + ".txt"
            summary_file = open(summary_path, "a")
            summary_file.write("{}: AUC_ROC: {}".format(task, auc_roc_val))
            summary_file.write("\n")
            summary_file.write("{}: AUC_PR: {}".format(task, auc_pr_val))
            summary_file.write("\n")
            summary_file.close()
    else:
        for (inp_val, tar_val, segment_label_val) in validation_ds:
            targets_val, predictions_val, weights_val = accuracy_step(inp_val, tar_val, segment_label_val)
            val_accuracy.update_state(targets_val, predictions_val, sample_weight = weights_val)

        print("Performance over validation: Accuracy {:.4f}".format(val_accuracy.result()))

        if summary_dir:
            summary_path = summary_dir + ".txt"
            summary_file = open(summary_path, "a")
            summary_file.write("{}: ACC: {}".format(task, val_accuracy.result()))
            summary_file.write("\n")
            summary_file.close()

    print ('Time taken training: {} secs\n'.format(time.time() - start))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
