import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score, roc_auc_score


class FromLogitsMixin:
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)


class AUC(FromLogitsMixin, tf.metrics.AUC):
    ...


class BinaryAccuracy(FromLogitsMixin, tf.metrics.BinaryAccuracy):
    ...


class TruePositives(FromLogitsMixin, tf.metrics.TruePositives):
    ...


class FalsePositives(FromLogitsMixin, tf.metrics.FalsePositives):
    ...


class TrueNegatives(FromLogitsMixin, tf.metrics.TrueNegatives):
    ...


class FalseNegatives(FromLogitsMixin, tf.metrics.FalseNegatives):
    ...


class Precision(FromLogitsMixin, tf.metrics.Precision):
    ...


class Recall(FromLogitsMixin, tf.metrics.Recall):
    ...


class F1Score(FromLogitsMixin, tfa.metrics.F1Score):
    ...


class Result:
    def __init__(self):
        self.accuracy_list = []
        self.sensitivity_list = []
        self.specificity_list = []
        self.f1_list = []
        self.auroc_list = []
        self.auprc_list = []
        self.precision_list = []

    def add(self, y_test, y_predict, y_score):
        C = confusion_matrix(y_test, y_predict, labels=(1, 0))
        TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]

        acc, sn, sp, pr = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP), 1. * TP / (
                TP + FP)
        f1 = f1_score(y_test, y_predict)
        auc = roc_auc_score(y_test, y_score)
        auprc = average_precision_score(y_test, y_score)

        self.accuracy_list.append(acc * 100)
        self.precision_list.append(pr * 100)
        self.sensitivity_list.append(sn * 100)
        self.specificity_list.append(sp * 100)
        self.f1_list.append(f1 * 100)
        self.auroc_list.append(auc * 100)
        self.auprc_list.append(auprc * 100)


    def get(self):
        out_str = "=========================================================================== \n"
        out_str += str(self.accuracy_list) + " \n"
        out_str += str(self.precision_list) + " \n"
        out_str += str(self.sensitivity_list) + " \n"
        out_str += str(self.specificity_list) + " \n"
        out_str += str(self.f1_list) + " \n"
        out_str += str(self.auroc_list) + " \n"
        out_str += str(self.auprc_list) + " \n"
        out_str += str("Accuracy: %.2f -+ %.3f" % (np.mean(self.accuracy_list), np.std(self.accuracy_list))) + " \n"
        out_str += str("Precision: %.2f -+ %.3f" % (np.mean(self.precision_list), np.std(self.precision_list))) + " \n"
        out_str += str(
            "Recall: %.2f -+ %.3f" % (np.mean(self.sensitivity_list), np.std(self.sensitivity_list))) + " \n"
        out_str += str(
            "Specifity: %.2f -+ %.3f" % (np.mean(self.specificity_list), np.std(self.specificity_list))) + " \n"
        out_str += str("F1: %.2f -+ %.3f" % (np.mean(self.f1_list), np.std(self.f1_list))) + " \n"
        out_str += str("AUROC: %.2f -+ %.3f" % (np.mean(self.auroc_list), np.std(self.auroc_list))) + " \n"
        out_str += str("AUPRC: %.2f -+ %.3f" % (np.mean(self.auprc_list), np.std(self.auprc_list))) + " \n"

        out_str += str("$ %.1f \pm %.1f$" % (np.mean(self.accuracy_list), np.std(self.accuracy_list))) + "& "
        out_str += str("$%.1f \pm %.1f$" % (np.mean(self.precision_list), np.std(self.precision_list))) + "& "
        out_str += str("$%.1f \pm %.1f$" % (np.mean(self.sensitivity_list), np.std(self.sensitivity_list))) + "& "
        out_str += str("$%.1f \pm %.1f$" % (np.mean(self.f1_list), np.std(self.f1_list))) + "& "
        out_str += str("$%.1f \pm %.1f$" % (np.mean(self.auroc_list), np.std(self.auroc_list))) + "& "

        return out_str

    def print(self):
        print(self.get())

    def save(self, path, config):
        file = open(path, "w+")
        file.write(str(config))
        file.write("\n")
        file.write(self.get())
        file.flush()
        file.close()



def NMSE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1) / (K.clip(K.mean(K.abs(y_true), axis=-1), K.epsilon(), None) * 2)
