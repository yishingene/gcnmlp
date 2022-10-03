import torch
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc, f1_score,precision_score, recall_score,average_precision_score,label_ranking_average_precision_score
from sklearn import metrics
import torch.nn.functional as F

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    prediction_int = np.zeros_like(scores)
    prediction_int[scores > 0.5] = 1
    f1_micro = f1_score(labels, prediction_int, average='micro',zero_division=0)
    f1_macro = f1_score(labels, prediction_int, average='macro',zero_division=0)
    f1_weight = f1_score(labels, prediction_int, average='weighted',zero_division=0)
    prec_micro = precision_score(labels, prediction_int, average='micro',zero_division=0)
    prec_macro = precision_score(labels, prediction_int, average='macro',zero_division=0)
    prec_weight = precision_score(labels, prediction_int, average='weighted',zero_division=0)
    recall_micro = recall_score(labels, prediction_int, average='micro',zero_division=0)
    recall_macro = recall_score(labels, prediction_int, average='macro',zero_division=0)
    recall_weight = recall_score(labels, prediction_int, average='weighted',zero_division=0)

    prec, recall, _ = precision_recall_curve(labels,scores)
    pr_auc_score = auc(recall, prec)

    fpr, tpr, _ = metrics.roc_curve(labels,scores)
    roc_auc_score = metrics.auc(fpr, tpr)
 
    return roc_auc_score, pr_auc_score, f1_micro, f1_macro, f1_weight, prec_micro, prec_macro, prec_weight,     recall_micro,recall_macro, recall_weight,prec, recall, fpr, tpr

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).squeeze(1)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)