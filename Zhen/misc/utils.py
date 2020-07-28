import os
import numpy as np
from time import time, sleep
from datetime import datetime
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    if self.log_file is None:
        os.mkdir(self.output_folder)
        timestamp = datetime.now()
        self.log_file = os.path.join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                     (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                      timestamp.second))

        with open(self.log_file, 'w') as f:
            f.write("Starting... \n")

    successful = False
    max_attempts = 5
    ctr = 0

    while not successful and ctr < max_attempts:
        try:
            with open(self.log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1

    if also_print_to_console:
        print(*args)


def get_pred_class_index(pred_score, threshold):
    pred_label = (pred_score > threshold).astype(np.int)

    pred_label = pred_label.squeeze()

    return pred_label


def calculate_metrics(pred_score, pred_label, target, multi_class=False):
    if multi_class:
        acc = metrics.accuracy_score(target, pred_label)

        lb = LabelBinarizer()
        lb.fit(target)

        target = lb.transform(target)  # N*classes
        pred_label = lb.transform(pred_label)

        auc = metrics.roc_auc_score(target, pred_score, average='macro')
        f1_score = metrics.f1_score(target, pred_label, average='macro')
        precision = metrics.precision_score(target, pred_label, average='macro')
        recall = metrics.recall_score(target, pred_label, average='macro')

    else:
        acc = metrics.accuracy_score(target, pred_label)
        auc = metrics.roc_auc_score(target, pred_score)
        f1_score = metrics.f1_score(target, pred_label)
        precision = metrics.precision_score(target, pred_label)
        recall = metrics.recall_score(target, pred_label)

    return acc, auc, f1_score, precision, recall
