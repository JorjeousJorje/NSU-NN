import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.where(prediction == ground_truth)[0].shape[0] / prediction.shape[0]
    return accuracy
