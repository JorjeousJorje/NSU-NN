import numpy as np
from collections import Counter

def binary_classification_metrics(prediction, ground_truth):
    expectations = Counter(zip(prediction, ground_truth))
    
    precision = expectations[(True, True)] / (expectations[(True, True)] + expectations[(True, False)])
    recall = expectations[(True, True)] / (expectations[(True, True)] + expectations[(False, True)])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (expectations[(True, True)] + expectations[(False, False)]) / sum(expectations.values())

    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.where(prediction == ground_truth)[0].shape[0] / prediction.shape[0]
    return accuracy


