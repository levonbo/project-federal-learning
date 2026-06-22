from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def getACC(y_true, y_score, task, threshold=0.5):
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret


def getAUC(y_true, y_score, task):
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()
    
    if task == "multi-label, binary-class":
        auc = 0
        valid = 0
        for i in range(y_score.shape[1]):
            y_true_binary = y_true[:, i]
            y_score_binary = y_score[:, i]
            if y_score_binary.sum() == 0:
                continue
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
            valid += 1
        ret = auc / valid
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        valid = 0 
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            if y_true_binary.sum() == 0:
                continue
            auc += roc_auc_score(y_true_binary, y_score_binary)
            valid += 1 
        ret = auc / valid

    return ret
