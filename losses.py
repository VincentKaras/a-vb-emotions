import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import CCC
from end2you.utils import Params
from typing import List

from dataset import VOCAL_TYPES, DIMENSIONS, EMOTIONS, CULTURE_EMOTIONS


"""
Loss definitions
"""

class CCC_Loss(nn.Module):
    """
    Computes CCC loss for multidimensional targets and averages / weighs the result
    """

    def __init__(self, num_targets:int, weight=None, reduction="mean") -> None:
        super().__init__()

        self.num_targets = num_targets
        self.reduction = reduction

        # ddruw
        self.phi = 1.0
        self.kappa = num_targets   # scales the sum of the dynamic weights
        self.temperature = 10   # configure later
        self.loss_t_1 = None
        self.loss_t_2 = None

       
        if self.reduction == "mean":
            print("\t* CCC loss module with mean reduction *")
        elif self.reduction == "rruw" or self.reduction == "druw": # uncertainty weighting
            #self.log_vars = nn.Parameter(torch.FloatTensor([1 / self.num_targets] * self.num_targets))
            self.log_vars = nn.Parameter(torch.ones(self.num_targets) * 1 / self.num_targets)
            if self.reduction == "rruw":
                print("\t* CCC loss module with restrained uncertainty weighting *")
            else:
                print("\t* CCC loss module with dynamic restrained uncertainty weighting *")
        elif self.reduction == "dwa": 
            print("\t* CCC loss module dynamic weight average *")
        else:
            raise NotImplementedError("Reduction method {} not implemented!".format(reduction))

        if weight is not None:
            self.weight == nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = None

    
    #def dynamic_weight_average(self, loss_t1, loss_t2) -> torch.FloatTensor: 
        """
        computes dynamic loss weight from the losses of the last two steps.
        Dynamic weights sum up to N
        :loss_t1 FloatTensor or list of floats of losses one step before
        :loss_t2 FloatTensor or list of floats of losses two steps before
        """

    #    if (loss_t1 is None) or (loss_t2 is None):  # if there are no previous time steps, return ones
    #        #return [1.0] * len(self.num_targets)
    #        return torch.ones(self.num_targets)
        
    #    assert len(loss_t1) == len(loss_t2), "Loss lists must have same number of tasks for each step"

    #    if isinstance(loss_t1, list):
    #        loss_t1 = torch.FloatTensor(loss_t1)
    #     if isinstance(loss_t2, list):
    #        loss_t2 = torch.FloatTensor(loss_t2)

    #    dl = loss_t1 / loss_t2
    #    # print(dl)
    #    dyn_weights = self.kappa * torch.softmax(dl / self.temperature, dim=0)

    #    return dyn_weights


    def forward(self, preds, targets):

        assert preds.shape == targets.shape

        if preds.dim() > 2:
            preds = preds.view(-1, preds.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

        # compute CCC
        ccc = CCC(preds, targets)
        ccc_losses = 1 - ccc

        # average the CCCs over the emotions (default) or compute weighted sum
        if self.reduction == "sum" and self.weight is not None:
            return  torch.sum(ccc_losses * self.weight)

        # uncertainty
        elif self.reduction == "rruw" or self.reduction == "druw":
            un_weights = 1 / (self.num_targets * self.log_vars ** 2)
            regularisation = torch.sum(torch.log(1 + self.log_vars ** 2))
            constraint = torch.abs(self.phi - torch.sum(torch.abs(self.log_vars)))

            if self.reduction == "rruw": 
                return torch.sum(un_weights * ccc_losses) + regularisation + constraint

            else: #druw
                # compute dynamic weights and move them to same device as losses
                dyn_weights = dynamic_weight_average(num_tasks=self.num_targets, kappa=self.kappa, temperature=self.temperature,loss_t1=self.loss_t_1, loss_t2=self.loss_t_2).to(ccc_losses.device)
                
                self.loss_t_2 = self.loss_t_1
                self.loss_t_1 = ccc_losses.detach() # remove gradient

                return torch.sum((dyn_weights + un_weights) * ccc_losses) + regularisation + constraint

        elif self.reduction == "dwa":
             # compute dynamic weights and move them to same device as losses
            dyn_weights = dynamic_weight_average(num_tasks=self.num_targets, kappa=self.kappa, temperature=self.temperature, loss_t1=self.loss_t_1, loss_t2=self.loss_t_2).to(ccc_losses.device)
                
            self.loss_t_2 = self.loss_t_1
            self.loss_t_1 = ccc_losses.detach() # remove gradient

            return torch.sum(dyn_weights * ccc_losses)


        else:   # default mean reduction
            return torch.mean(ccc_losses)
        
        #loss = 1 - ccc

        # return loss

# TODO Uncertainty Losses


class Criterion(nn.Module):
    """
    criterion definition for training. Incorporates multiple losses
    """
    def __init__(self, params:Params) -> None:
        super().__init__()
        
        self.tasks = ["voc_type", "low", "high", "culture_emotion"]
        self.tasks_dict = {
            "voc_type": {
                "type": "classification",
            },
            "low": {
                "type": "regression",
                "dimensions": DIMENSIONS,
            },
            "high": {
                "type": "regression",
                "dimensions": EMOTIONS,
            },
            "culture_emotion": {
                "type": "regression",
                "dimensions": CULTURE_EMOTIONS,
            }

        }
        self.params = params
        loss_strategy = params.train.loss_strategy

        # create loss modules
        loss_dict = {}
        for t in self.tasks:
            if self.tasks_dict[t]["type"] == "classification":
                loss_dict[t] = nn.CrossEntropyLoss()
                print("\t* Crossentropy Loss *")
            else:
                loss_dict[t] = CCC_Loss(num_targets=len(self.tasks_dict[t]["dimensions"]), reduction=loss_strategy)
        # put losses in a module dict so they will be registered properly
        self.losses = nn.ModuleDict(loss_dict)

        #self.classification_loss = nn.CrossEntropyLoss()
        #self.regression_loss = CCC_Loss()   # make this configurable later

    
    def forward(self, preds, targets, return_all=True):
        """
        Basic additive loss. Returns mean of the losses, plus optional a dict of individual task results
        """
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task]

            #if task in ["voc_type"]:
            #    task_loss =  self.classification_loss(pred, target)
            #else:   # regression
            #    task_loss = self.regression_loss(pred, target)

            #task_loss = self.losses[task](pred, target)
            task_losses.append(self.losses[task](pred, target))

            #loss += task_loss
            #if return_all:
            #    task_losses[task] = task_loss.item()    # update dict with value of single loss

        #if isinstance(pred, dict) and isinstance(target, dict):
        loss = torch.mean(torch.stack(task_losses))

        #loss /= len(self.tasks) # divide by number of tasks

        if return_all:
            return loss, {t: task_losses[i].item() for i,t in enumerate(self.tasks)}

        else: 
            return loss


class UncertaintyCriterion(Criterion):
    """
    Criterion definition which adds restrained uncertainty to weigh the losses.
    """

    def __init__(self, params: Params) -> None:
        super().__init__(params)
        # init parameters
        self.log_vars = nn.Parameter(torch.FloatTensor([1/len(self.tasks)] * len(self.tasks)))
        # constraint value
        self.phi = 1.0

    
    def forward(self, preds, targets, return_all=True):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task]

            task_losses.append(self.losses[task](pred, target))
            #if return_all:
            #    task_losses[task] = task_loss.item()    # update dict with value of single loss

            #task_loss = 1 / (len(self.tasks) * self.log_vars[task_index] ** 2) * task_loss + torch.log(1 + self.log_vars[task_index] ** 2)

            #loss += task_loss

        loss = torch.stack(task_losses)

        un_weights = 1 / (len(self.tasks) * self.log_vars ** 2)
        regularisation = torch.sum(torch.log(1 + self.log_vars ** 2))
        constraint = torch.abs(self.phi - torch.sum(torch.abs(self.log_vars)))

        loss = torch.sum(un_weights * loss) + regularisation + constraint

        if return_all:
            return loss, {t: task_losses[i].item() for i,t in enumerate(self.tasks)}
        else:
            return loss

class DynamicUncertaintyCriterion(Criterion):
    """
    Criterion definition which implements Dynamic Restrained Uncertainty Weighting for losses.
    See Song et al. (2022)
    """

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        N = len(self.tasks)

        # init parameters
        self.log_vars = nn.Parameter(torch.ones(N) * 1 / N)
      
        self.phi = 1.0    # constraint value
        self.kappa = N   # scales the dynamic weights
        self.temperature = 10   # default value for smoothing softmax

        self.loss_t_1 = None
        self.loss_t_2 = None

    
    #def dynamic_weight_average(self, loss_t1, loss_t2) -> torch.FloatTensor:
        """
        computes dynamic loss weight from the losses of the last two steps 
        """

    #    N = len(self.tasks)

    #    if (loss_t1 is None) or (loss_t2 is None):  # if there are no previous time steps, return ones
    #        return torch.ones(N)
        
    #    assert len(loss_t1) == len(loss_t2), "Loss lists must have same number of tasks for each step"

    #    if isinstance(loss_t1, list):
    #        loss_t1 = torch.FloatTensor(loss_t1)
    #    if isinstance(loss_t2, list):
    #        loss_t2 = torch.FloatTensor(loss_t2)

    #   dl = loss_t1 / loss_t2
    #    dyn_weights = self.kappa * torch.softmax(dl / self.temperature, dim=0)

    #    assert isinstance(dyn_weights, torch.Tensor), "weights should be a Tensor but is {}".format(type(dyn_weights))

    #    return dyn_weights


    def forward(self, preds, targets, return_all=True):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task_index]

            task_losses.append(self.losses[task](pred, target))

            #task_loss = self.losses[task](pred, target)
            #if return_all:
            #    task_losses[task] = task_loss.item()    # update dict with value of single loss

        loss_t = torch.stack(task_losses)   # combine N 0-dimensional tensors into one tensor of size [N,]
        
        dyn_weights = dynamic_weight_average(num_tasks=len(self.tasks), kappa=self.kappa, temperature=self.temperature, loss_t1=self.loss_t_1, loss_t2=self.loss_t_2)
        dyn_weights = dyn_weights.to(loss_t.device) # move weights to the same device the loss tensor is on
        un_weights = 1 / (len(self.tasks) * self.log_vars ** 2)
        regularisation = torch.sum(torch.log(1 + self.log_vars ** 2))
        constraint = torch.abs(self.phi - torch.sum(torch.abs(self.log_vars)))
        
        loss = torch.sum((dyn_weights + un_weights) * loss_t) + regularisation + constraint

        # update states
        self.loss_t_2 = self.loss_t_1
        self.loss_t_1 = loss_t.detach()

        if return_all:
            return loss, {t: task_losses[i].item() for i,t in enumerate(self.tasks)}
        else:
            return loss


class DynamicWeightAverageCriterion(Criterion):
    def __init__(self, params: Params) -> None:
        super().__init__(params)

        self.kappa = len(self.tasks)
        self.temperature = 10.0

        self.loss_t_1 = None
        self.loss_t_2 = None


    def forward(self, preds, targets, return_all=True):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):
            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task_index]

            task_losses.append(self.losses[task](pred, target))

        loss_t = torch.stack(task_losses)

        dyn_weights = dynamic_weight_average(num_tasks=len(self.tasks), kappa=self.kappa, temperature=self.temperature, loss_t1=self.loss_t_1, loss_t2=self.loss_t_2)
        dyn_weights = dyn_weights.to(loss_t.device)
        loss = torch.sum(dyn_weights * loss_t)

        # update states
        self.loss_t_2 = self.loss_t_1
        self.loss_t_1 = loss_t.detach()

        if return_all:
            return loss, {t: task_losses[i].item() for i,t in enumerate(self.tasks)}
        else:
            return loss




def criterion_factory(params:Params) -> Criterion:
    """
    factory helper that generates the desired criterion based on the loss_strategy flag
    :params Params object, containing train.loss_strategy str, one of [mean, rruw, druw]
    """ 

    loss_strategy = params.train.loss_strategy

    if loss_strategy == "mean":
        print("Averaging task losses")
        return Criterion(params=params)
    elif loss_strategy == "rruw":
        print("Using Restrained Revised Uncertainty Weighting for losses")
        return UncertaintyCriterion(params=params)
    elif loss_strategy == "druw":
        print("Using Dynamic Restrained Uncertainty Weighting for losses")
        return DynamicUncertaintyCriterion(params=params)
    elif loss_strategy == "dwa":
        print("Using Dynamic Weight Averaging for losses")
        return DynamicWeightAverageCriterion(params=params)
    else: 
        raise NotImplementedError("{} not implemented".format(loss_strategy))

    
def dynamic_weight_average(num_tasks:int, kappa:float, temperature:float, loss_t1, loss_t2) -> torch.FloatTensor:
        """
        computes dynamic loss weight from the losses of the last two steps 
        """

        if (loss_t1 is None) or (loss_t2 is None):  # if there are no previous time steps, return ones
            return torch.ones(num_tasks)
        
        assert len(loss_t1) == len(loss_t2), "Loss lists must have same number of tasks for each step"

        if isinstance(loss_t1, list):
            loss_t1 = torch.FloatTensor(loss_t1)
        if isinstance(loss_t2, list):
            loss_t2 = torch.FloatTensor(loss_t2)

        dl = loss_t1 / loss_t2
        dyn_weights = kappa * torch.softmax(dl / temperature, dim=0)

        assert isinstance(dyn_weights, torch.Tensor), "weights should be a Tensor but is {}".format(type(dyn_weights))

        return dyn_weights



