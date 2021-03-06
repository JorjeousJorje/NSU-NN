from collections import Counter
import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    
    # zip forms tuples (bool, bool)
    # counts tells us about TP, FP, FN, TN (gives set of tuples)
    expectations = Counter(zip(prediction, ground_truth))
    
    precision = expectations[(True, True)] / (expectations[(True, True)] + expectations[(True, False)])
    
    recall = expectations[(True, True)] / (expectations[(True, True)] + expectations[(False, True)])
    
    f1 = 2 * precision * recall / (precision + recall)
    
    accuracy = (expectations[(True, True)] + expectations[(False, False)]) / sum(expectations.values())

    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    # how many samples are correctly predicted:
    #  -> np.where(prediction == ground_truth)[0].shape[0]
    # whole prediction:
    #  -> prediction.shape[0]
    accuracy = np.where(prediction == ground_truth)[0].shape[0] / prediction.shape[0]
    return accuracy
