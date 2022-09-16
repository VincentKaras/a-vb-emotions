"""
Metrics definitions

"""

import torch
import torch.nn.functional as F
import numpy as np
from end2you.utils import Params
from sklearn.metrics import recall_score
from typing import Any, Dict
from dataset import VOCAL_TYPES, DIMENSIONS, CULTURES, EMOTIONS, CULTURE_EMOTIONS

def CCC(preds:torch.Tensor, targets:torch.Tensor):
    """
    Compute the CCC
    preds:  tensor of shape [B, N], where B is batch size and N is the number of emotions evaluated
    targets: tensor [B, N]
    returns N values of CCC, one per dimension/category of emotion
    """

    # mean and var over the batch dimension
    preds_mean = torch.mean(preds, 0)   # [N]
    targets_mean = torch.mean(targets, 0) # [N]
    preds_var = torch.var(preds, 0) #  [N]
    targets_var = torch.var(targets, 0) # [N]

    # 
    cov = torch.mean((preds - preds_mean) * (targets - targets_mean), 0)    # [N]
    ccc = 2 * cov / (preds_var + targets_var + torch.square(preds_mean - targets_mean)) # [N]

    return ccc


def np_CCC(preds:np.ndarray, targets:np.ndarray):
    """
    Compute the CCC
    
    """

    preds_mean = preds.mean(axis=0)
    targets_mean = targets.mean(axis=0)

    preds_var = preds.var(axis=0)
    targets_var = targets.var(axis=0)

    cov = np.mean((preds - preds_mean) * (targets - targets_mean), axis=0)
    ccc = 2 * cov / (preds_var + targets_var + (preds_mean - targets_mean) ** 2)

    return ccc


def pearson_correlation_coefficient(preds:np.ndarray, targets:np.ndarray):
    """
    Compute the Pearson Correlation coefficient
    """

    # pc = cov(x, y) / (var(x) * var(y))

    preds_mean = preds.mean(axis=0)
    targets_mean = targets.mean(axis=0)

    preds_std = preds.std(axis=0)
    targets_std = targets.std(axis=0)

    cov = np.mean((preds - preds_mean) * (targets - targets_mean), axis=0)
    pearson = cov / (preds_std * targets_std)

    return pearson


def UAR(preds:torch.Tensor, targets:torch.Tensor):
    """
    Compute the unweighted average recall. Wraps around sklearn.metric.recall_score
    preds: tensor of shape [B, N] where N is the number of classes. It is expected that this is before softmax
    targets: tensor of shape [B]
    """

    if preds.dim() > 1:
        preds = torch.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)

    preds = preds.squeeze()
    targets = targets.squeeze()

    assert preds.size() == targets.size(), "Sizes of predictions {} and targets {} do not match".format(preds.size(), targets.size()) 

    # compute recall unweighted
    return recall_score(y_true=targets.numpy(), y_pred=preds.numpy(), average="macro")


def np_UAR(preds:np.ndarray, targets:np.ndarray):
    """
    Compute unweighted average recall. wraps around sklearn
    """

    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)
    
    preds = preds.squeeze()
    targets = targets.squeeze()

    assert preds.shape == targets.shape, "Size of predictions {} and targets {} must match".format(preds.shape, targets.shape)

    return recall_score(y_true=targets, y_pred=preds, average="macro")


class Metric():
    """
    helper object which aggregates metrics for evaluation
    """

    def __init__(self, params:Params=None) -> None:
        
        self.params = params

        self.tasks = ["voc_type", "low", "high", "culture_emotion"]
 
        self.classification_metrics = {"UAR": UAR}
        self.regression_metrics = {"Pearson": pearson_correlation_coefficient,
                                    "CCC": np_CCC}                                       # use np_CCC implementation to get back a numpy array. Or use CCC and cast

        # this could also be delivered via the params object
        self.tasks_dict  = {
            "voc_type": {
                "type": "classification",
                "categories": VOCAL_TYPES,
                "metrics": ["UAR"],
                "score": "UAR",
            },
            "low": {
                "type": "regression",
                "dimensions": DIMENSIONS,
                "metrics": ["CCC", "Pearson"],
                "score": "CCC",
            },
            "high": {
                "type": "regression",
                "dimensions": EMOTIONS,
                "metrics": ["CCC", "Pearson"],
                "score": "CCC",
            },
            "culture_emotion": {
                "type": "regression",
                "dimensions": CULTURE_EMOTIONS,
                "metrics": ["CCC", "Pearson"],
                "score": "CCC",
            }

        }


    def compute(self, preds:Dict[str, torch.Tensor], targets:Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Dict[str, np.float64]]]:
        """
        compute appropriate metric for each task
        returns a nested dict with level task, metric, dimension.
        Metric values are cast to np.float64 for later json serialization
        """
        metrics = {}
        for t in self.tasks:
            metrics[t] = {}

            #if t in ["voc_type"]:
            if self.tasks_dict[t]["type"] == "classification":
                for m in self.tasks_dict[t]["metrics"]:        # iterate over all classification metrics
                    metrics[t][m] = {}
                    result = np.float64(self.classification_metrics[m](preds=preds[t], targets=targets[t]))
                    metrics[t][m]["all"] = result   # overall result
                    # add another entry here for logging 
                    metrics[t][m][t] = result

            else:   # regression
                for m in self.tasks_dict[t]["metrics"]:
                    metrics[t][m] = {}
                    result = np.float64(self.regression_metrics[m](preds=preds[t].numpy(), targets=targets[t].numpy()))  # numpy array of DIM values
                    metrics[t][m]["all"] = np.mean(result)
                    for i, d in enumerate(self.tasks_dict[t]["dimensions"]):    # assume order is always correct
                        metrics[t][m][str(d)] = result[i]

        return metrics


