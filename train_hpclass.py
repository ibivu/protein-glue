import tensorflow as tf
import time
import click
from tensorflow_addons.optimizers import LAMB

from dataset.HPC import create_dataset
from model import BERTTransformer, HPCClassifier

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

import numpy as np

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_function(real, pred, loss_weights):
    loss_ = loss_object(real, pred)
    loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)
    loss_ *= loss_weights

    return tf.reduce_sum(loss_)/tf.reduce_sum(loss_weights)

# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_precision = tf.keras.metrics.Precision(name='train_precision')
train_AUC = tf.keras.metrics.AUC(name = 'train_AUC')
train_recall = tf.keras.metrics.Recall(name='train_recall')
train_FP = tf.keras.metrics.FalsePositives(name='train_FP')
train_TP = tf.keras.metrics.TruePositives(name = 'train_TP')
train_FN = tf.keras.metrics.FalseNegatives(name = 'train_FN')
train_TN = tf.keras.metrics.TrueNegatives(name = 'train_TN')

val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
val_FP = tf.keras.metrics.FalsePositives(name='val_FP')
val_TP = tf.keras.metrics.TruePositives(name = 'val_TP')
val_FN = tf.keras.metrics.FalseNegatives(name = 'val_FN')
val_TN = tf.keras.metrics.TrueNegatives(name = 'val_TN')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

# acc_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#     tf.TensorSpec(shape=None, dtype=tf.float32)
# ]

POSITIONAL_ENCODING_MAX_LENGTH = 1700
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

        #do not update accuracy here because 2 accuracy metrics and if statement not possible

        #Needed for curves
        interface_prediction = predictions[:,:,2]
        if_pred_1D = tf.reshape(interface_prediction,[-1])
        tar_1D = tf.reshape(tar, [-1])
        predictions_max = tf.argmax(predictions, axis = 2)
        predictions_max_1D = tf.reshape(predictions_max, [-1])

        #return tar, predictions, loss_weights
        return tar, predictions, loss_weights, tar_1D, predictions_max_1D, if_pred_1D

    def correct_formatting(tar_list, pred_max_list, if_pred_list):
        tar_list_np = tar_list.numpy()
        IF_pred_list_np = if_pred_list.numpy()
        IF_pred_max_list_np = pred_max_list.numpy()

        index_no_aa = [i for i, x in enumerate(tar_list_np) if x == 0]
        tar_list_np_filtered = [i for j, i in enumerate(tar_list_np) if j not in set(index_no_aa)]
        IF_pred_list_np_filtered = [i for j, i in enumerate(IF_pred_list_np) if j not in set(index_no_aa)]
        IF_pred_list_max_np_filtered = [i for j, i in enumerate(IF_pred_max_list_np) if j not in set(index_no_aa)]

        target_correct = [0 if x==1 else x for x in tar_list_np_filtered]
        target_correct[:] =  [1 if x==2 else x for x in target_correct]
        pred_prob_correct = IF_pred_list_np_filtered
        predict_correct = [0 if x==1 else x for x in IF_pred_list_max_np_filtered]
        predict_correct[:] =  [1 if x==2 else x for x in predict_correct]

        return target_correct, predict_correct, pred_prob_correct


    def metrices(targets, predictions, predictions_probabilities_interface, set):
        fpr , tpr , thresholds = roc_curve(targets, predictions_probabilities_interface)
        auc_score = roc_auc_score(targets,predictions_probabilities_interface)
        precision = precision_score(targets, predictions, zero_division = 0)
        recall = recall_score(targets, predictions, zero_division = 0)
        precision_list , recall_list , thresholds_PR = precision_recall_curve(targets, predictions_probabilities_interface)

        P = sum(targets)
        N = len(targets)-P
        fraction_positive = P / (P+N)

        if set == "val":
            val_FP(targets, predictions)
            val_TP(targets, predictions)
            val_FN(targets, predictions)
            val_TN(targets, predictions)
        else:
            train_precision(targets, predictions)
            train_recall(targets, predictions)
            train_AUC(targets, predictions)
            train_TP(targets, predictions)
            train_FP(targets, predictions)
            train_FN (targets, predictions)
            train_TN(targets, predictions)

        return fpr, tpr, auc_score, precision, recall, precision_list, recall_list, fraction_positive

    def plotting_roc(fpr, tpr, auc_score, path):
        lw = 1
        plt.figure(1)           #write all ROC to this output file
        plt.plot(fpr, tpr, label= 'AUC = {:.4f}'.format(auc_score))
        plt.plot([0,1], [0,1], color="navy", lw=lw, linestyle='--')
        plt.ylabel('True positive rate', size = 10)
        plt.xlabel("False positive rate", size = 10)
        plt.legend(loc="lower right", fontsize= 8)
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.savefig(path)
        #plt.close()        ##write all seperate roc plots

    def plotting_pr(recall, precision, fraction_positive, path):
        lw = 1
        plt.figure(2)       ##write all PR plots to same file
        plt.plot(recall, precision, label = 'PR')
        plt.hlines(fraction_positive, 0, 1, color="navy", lw=lw, linestyle='--')
        plt.ylabel('precision', size = 10)
        plt.xlabel("recall", size = 10)
        plt.legend(loc="upper right", fontsize= 8)
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.savefig(path)
        #plt.close()        #write all PR plots to different files

    def write_output_plots(auc_score, fraction_positive, targets, predictions, pred_prob_IF, path):
        file = open(path, "w")
        file.write("auc_score: " + str(auc_score) + "\n")
        file.write("fraction_positive: " + str(fraction_positive) + "\n")
        file.write("targets: " + str(targets) + "\n")
        file.write("predictions: " + str(predictions) + "\n")
        file.write("pred_prob_IF: " + str(pred_prob_IF) + "\n")
        file.close()


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
    classifier = HPCClassifier(dff=d_ff, rate=dropout_rate)

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
                #tf.print("validation")
                now = time.time()
                seqs_per_sec = (batch_size * LOGGING_EVERY_STEPS) / (now - start_batch)
                start_batch = now

                try:
                    accuracy_inp, accuracy_tar = next(validation_iter) if validation_iter else (inp, tar)
                except StopIteration:
                    validation_iter = iter(validation_ds) if validation_ds else None
                    accuracy_inp, accuracy_tar = next(validation_iter) if validation_iter else (inp, tar)

                # targets, predictions, weights = accuracy_step(accuracy_inp, accuracy_tar)
                # train_accuracy(targets, predictions, sample_weight = weights)
                tar, predictions, loss_weights, tar_list, pred_max_1D, IF_pred_list = accuracy_step(accuracy_inp, accuracy_tar)
                train_accuracy(tar, predictions, sample_weight = loss_weights)
                targets, predictions, pred_prob_IF = correct_formatting(tar_list, pred_max_1D, IF_pred_list)
                fpr, tpr, auc_score, precision, recall, precision_list, recall_list, fraction_positive = metrices(targets, predictions, pred_prob_IF, "train")

                print ('Steps {} (Epoch {} Batch {}) Seqs/sec {:.1f} Accuracy {:.4f} AUC {:.4f} Precision {:.4f} Recall {:.4f} TP {:.1f} FP {:.1f} TN {:.1f} FN {:.1f}'.format(
                    cur_step, epoch + 1, batch, seqs_per_sec, train_accuracy.result(), auc_score, precision, recall, train_TP.result(), train_FP.result(), train_TN.result(), train_FN.result()))

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
    # for (inp_val, tar_val) in validation_ds:
    #     targets_val, predictions_val, weights_val = accuracy_step(inp_val, tar_val)
    #     val_accuracy.update_state(targets_val, predictions_val, sample_weight = weights_val)
    #
    # print("Performance over validation: Accuracy {:.4f}".format(val_accuracy.result()))

    tar_list_val = []
    pred_list_val = []
    prob_list_val = []

    for (inp_val, tar_val) in validation_ds:
        tar, predictions, loss_weights, tar_list, pred_list, prob_list = accuracy_step(inp_val, tar_val)
        val_accuracy(tar, predictions, sample_weight=loss_weights)
        targets, predictions, pred_prob_IF = correct_formatting(tar_list, pred_list, prob_list)
        tar_list_val.extend(targets)
        pred_list_val.extend(predictions)
        prob_list_val.extend(pred_prob_IF)

    #After all validation data has been seen.
    fpr_val, tpr_val, auc_score_val, precision_val, recall_val, precision_list_val, recall_list_val, fraction_positive_val = metrices(tar_list_val, pred_list_val, prob_list_val, "val")

    print("creating plots")
    path_fig_roc = tensorboard_dir + "/ROC.png"
    path_fig_pr = tensorboard_dir + "/PR.png"
    path_output_write = tensorboard_dir + "/output.txt"

    plotting_roc(fpr_val, tpr_val, auc_score_val, path_fig_roc)
    plotting_pr(recall_list_val, precision_list_val, fraction_positive_val, path_fig_pr)
    write_output_plots(auc_score_val, fraction_positive_val, tar_list_val, pred_list_val, prob_list_val, path_output_write)

    print("Performance over validation: Accuracy {:.4f} AUC {:.4f} Precision {:.4f} Recall {:.4f} TP {:.1f} FP {:.1f} TN {:.1f} FN {:.1f}".format(val_accuracy.result(), auc_score_val, precision_val, recall_val, val_TP.result(), val_FP.result(), val_TN.result(), val_FN.result()))
    print ('Time taken training: {} secs\n'.format(time.time() - start))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()