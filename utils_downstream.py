"""
This script contains functions used in train_downstream.py
"""
import tensorflow as tf
import time
import click
from tensorflow_addons.optimizers import LAMB

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt

def performance_step_regression(targets, predictions):
    targets_ = tf.reshape(targets, [tf.shape(targets)[0]*tf.shape(targets)[1]])
    predictions_ = tf.reshape(predictions, [tf.shape(predictions)[0]*tf.shape(predictions)[1]])

    sample_weight = tf.cast(tf.math.greater(targets, 0), dtype=targets.dtype)
    sample_weight_ = tf.reshape(sample_weight, [tf.shape(sample_weight)[0]*tf.shape(sample_weight)[1]])
    indices = tf.squeeze(tf.where(tf.math.not_equal(sample_weight_, 0)), 1)
    tar = tf.gather(targets_, indices)
    pred = tf.gather(predictions_, indices)

    return tar, pred

def performance_step_interface(targets, predictions):
    interface_prediction = predictions[:,:,2]
    if_pred_1D = tf.reshape(interface_prediction,[-1])
    tar_1D = tf.reshape(targets, [-1])
    predictions_max = tf.argmax(predictions, axis = 2)
    predictions_max_1D = tf.reshape(predictions_max, [-1])

    return tar_1D, predictions_max_1D, if_pred_1D

def correct_formatting_interface(tar_list, pred_max_list, if_pred_list):
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


def metrices_interface(targets, predictions, predictions_probabilities_interface):
    fpr , tpr , thresholds = roc_curve(targets, predictions_probabilities_interface)
    auc_score = roc_auc_score(targets,predictions_probabilities_interface)
    precision = precision_score(targets, predictions, zero_division = 0)
    recall = recall_score(targets, predictions, zero_division = 0)
    precision_list , recall_list , thresholds_PR = precision_recall_curve(targets, predictions_probabilities_interface)
    auc_pr = auc(recall_list, precision_list)

    P = sum(targets)
    N = len(targets)-P
    fraction_positive = P / (P+N)

    return fpr, tpr, auc_score, auc_pr, precision, recall, precision_list, recall_list, fraction_positive

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

def write_output_plots(auc_roc, auc_pr, fraction_positive, targets, predictions, pred_prob_IF, path):
    file = open(path, "w")
    file.write("auc_roc: " + str(auc_roc) + "\n")
    file.write("auc_pr: " + str(auc_pr) + "\n")
    file.write("fraction_positive: " + str(fraction_positive) + "\n")
    file.write("targets: " + str(targets) + "\n")
    file.write("predictions: " + str(predictions) + "\n")
    file.write("pred_prob_IF: " + str(pred_prob_IF) + "\n")
    file.close()
