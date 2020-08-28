import numpy as np


def accuracy(ground_truth, predictions):
    return np.mean(ground_truth == predictions)


def confusion_matrix(ground_truth, predictions):
    classes = np.unique(ground_truth)
    confusion = np.zeros((len(classes), len(classes)))
    for i, prediction in enumerate(predictions):
        confusion[ground_truth[i], prediction] += 1
    return confusion
