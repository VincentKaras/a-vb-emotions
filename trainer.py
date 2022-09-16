"""
Trainer Class collecting components
"""

from asyncio import tasks
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import random
from end2you.utils import Params
from pathlib import Path
import json
from shutil import copy
from copy import deepcopy
from typing import Dict, Tuple

from dataset import CULTURES, EMOTIONS, DIMENSIONS, VOCAL_TYPES
import dataset
from dataset.vocal_data import VocalDataModule
from model.models import model_factory, count_all_parameters, count_trainable_parameters
from optimizer import get_optimizer, get_scheduler
from losses import Criterion, criterion_factory
from metrics import Metric

class Trainer():

    def __init__(self, params:Params) -> None:
        
        if params.model.pretrained_path:
            print("Pre-trained model passed!")
            ckpt = torch.load(str(params.model.pretrained_path))
            if "params" in ckpt.keys():
                print("Checkpoint contains saved params, restoring ...")
                saved_params = ckpt["params"]
                self.model_args = saved_params.model
                self.train_params = saved_params.train
                self.val_params = saved_params.val
                self.test_params = saved_params.test
                self.params = saved_params
            else:
                print("No saved params in checkpoint, falling back to CLI args ...")
                self.train_params = params.train # use the cli params to instantiate dataloaders
                self.val_params = params.val
                self.test_params = params.test
                self.model_args = params.model
                self.params = params
        else:   # no pretrained model checkpoint
            self.model_args = params.model    # use the cli args to instantiate the model 
            self.train_params = params.train # use the cli params to instantiate dataloaders
            self.val_params = params.val
            self.test_params = params.test
            self.params = params

        # tasks

        # TODO configurable later
        self.tasks = ["voc_type", "low", "high", "culture_emotion"]

        # create a model
        print("\n * * * Creating model * * * \n") 

        self.model = model_factory(self.model_args)
        if params.model.pretrained_path:    # if available, load pretrained weights
            if "state_dict" in ckpt.keys(): 
                self.model.load_state_dict(ckpt["state_dict"])
        # move model to GPU
        self.model.cuda()

        print("Model has {} parameters, {} of which are trainable".format(count_all_parameters(self.model), count_trainable_parameters(self.model)))

        print("Model setup complete!")

        # set the paths
        self.ckpt_dir = Path(self.params.checkpoints_dir)
        self.log_dir = Path(self.params.log_dir)
        self.root_dir = Path(self.params.root_dir)

        # seeds
        torch.manual_seed(params.train.seed)
        np.random.seed(params.train.seed)
        random.seed(params.train.seed)
        
        # create datasets
        print("\n* * * Creating dataloaders * * * \n")
        # lightning module
        dm = VocalDataModule(params)
        self.train_loader = dm.train_dataloader()
        self.val_loader = dm.val_dataloader()
        self.test_loader = dm.test_dataloader()

        print("Loader setup finished!")

        # TODO set up train loop scoring, saving, testing

        # allocate scores

        self.best_score = -1000
        self.best_scores = {}
        self.best_epoch = 0
        self.best_state = None
        self.best_path = Path("")
        self.best_results = None

        # improvement counter for early stopping
        self.no_improvement = 0

        # summary writer
        print("\n * * * Creating Tensorboard writer * * * \n")
        self.writer = SummaryWriter(log_dir=self.log_dir)    # multiples ? 

        # loss via criterion (multitask)
        print("\n * * * Setting up loss criterion * * * \n")
        #self.criterion = Criterion(params=params)
        self.criterion = criterion_factory(params=params)
        # if running on GPU, move criterion there
        if params.cuda:
            self.criterion.cuda()

        # optimizer and scheduler
        print("\n * * * Setting up optimizer and scheduler * * * \n")
        self.optimizer = get_optimizer(self.train_params, self.model, self.criterion)
        self.lr_scheduler = get_scheduler(self.optimizer)

        # metrics
        print("* * * Setting up the metrics * * * \n")
        self.metric = Metric(params=params)
    
        print("\n * * * * * Setup complete! * * * * * \n")
    
    ################################################################

    def train(self):
        """
        Training Loop
        """

        print("\n * * * Starting training * * * \n")

        epochs = self.train_params.num_epochs

        for epoch in range(1, epochs + 1):

            self.no_improvement += 1    # increment counter by 1 
            
            train_loss = 0.0
            # individual tasks tracker
            train_loss_tasks = {t: 0.0 for t in self.tasks}

            all_preds = {t: [] for t in self.tasks}
            all_labels = {t: [] for t in self.tasks}

            self.model.train()
            # iterate train loader
            for index, batch in enumerate(self.train_loader):
                # fwd pass
                audio = batch["audio"].cuda()
                #package labels
                labels = {}
                for task in self.tasks:
                    labels.update({task: batch[task].cuda()})
                    all_labels[task].append(batch[task])


                out = self.model(audio, labels)

                loss, task_losses = self.criterion.forward(out, labels, return_all=True)  # process these dicts
                
                self.optimizer.zero_grad()
                loss.backward() # bwd pass
                self.optimizer.step()

                for task in self.tasks:
                    all_preds[task].append(out[task].detach().cpu())

                train_loss += loss.item()   # add up losses per batch
                for t in self.tasks:
                    train_loss_tasks[t] += task_losses[t]

                # summary writer
                if index % self.params.train.save_summary_steps == 0:
                    self.writer.add_scalar(
                        "train/loss",
                        loss.item(),
                        global_step= (epoch -1) * len(self.train_loader) + index    # global index across epochs
                    )
                    print("\t Step {}/{} train loss: {:.3f}".format(index, len(self.train_loader), loss.item()))

            # mean across batches
            train_loss /= len(self.train_loader)
            for t in self.tasks:
                train_loss_tasks[t] = train_loss_tasks[t] / len(self.train_loader)

            for t in self.tasks:
                all_preds[t] = torch.cat(all_preds[t], dim=0)
                all_labels[t] = torch.cat(all_labels[t], dim=0)

            # calc train metrics
            train_metrics = self.metric.compute(preds=all_preds, targets=all_labels)
            train_summary_str = "Epoch {}/{} training metrics: ".format(epoch, epochs)
            
            # print out
            print("Epoch {}/{} train loss: {}".format(epoch, epochs, train_loss))
            self._print_metrics(train_summary_str, train_metrics)

            # dev evaluate
            print("\n * Validation step *")

            val_score, metrics, predictions, targets = self.evaluate("val")

            # summary writer
            self.writer.add_scalar(
                "val/score",
                val_score,
                epoch * len(self.train_loader)  # log one value at global index end of each train epoch
            )
            # get the individual values

            val_summary_str = "Epoch {}/{} validation metrics: ".format(epoch, epochs)
            self._print_metrics(val_summary_str, metrics=metrics)
            
            #val_task_str = {t: val_summary_str for t in self.tasks}

            #for t in self.tasks:
            #    task_info = self.metric.tasks_dict[t]
            #    m = task_info["score"]
            #    """
            #    if task_info["type"] == "classification":
            #        score = metrics[t].get(m)
            #        if score:
            #            val_summary_str = val_summary_str + " {} {}: {:.3f}".format(t, m, score["all"])
            #            val_task_str[t] = val_task_str[t]
            #    else:
            #        score = metrics[t].get(m)# is a numpy array 
            #        # val_task_str = "Epoch {}/{}".format(epoch, epochs) 
            #        if score:
            #            for i,d in enumerate(task_info["dimensions"]):
            #                val_task_str[t] = val_task_str[t] + " {} CCC: {:.3f}".format(d, score[i])
            #            val_summary_str = val_summary_str + " {} UAR: {:.3f}".format(t, score["all"])
            #    """

            #    result = metrics[t].get(m)
            #    if result:
            #        val_summary_str = val_summary_str + " {} {}: {:.3f}, ".format(t, m, result["all"]) # update score info
                    # add all other keys
            #        for k in result.keys():
            #            if k == "all":  # not needed here
            #                continue
            #            val_task_str[t] = val_task_str[t] + "{} {} : {:.3f}, ".format(k, m, result[k]) 

            # summary writer
            for t in self.tasks:
                m = self.metric.tasks_dict[t]["score"]
            self.writer.add_scalars(
                "val/{}/{}".format(t, m),
                metrics[t][m],
                epoch * len(self.train_loader)
            )
                        
            # print the info
            #print(val_summary_str)
            #for k, s in val_task_str.items():
            #    print(s)

                
            #for t in self.tasks:

            # update best results
            self._log_results(epoch, validation_score=val_score, metrics=metrics)
            
            # step the scheduler
            self.lr_scheduler.step(val_score)

        # final output
        print("\n * * * Overall best validation results at epoch {}/{}: {:.6f} * * * \n".format(self.best_epoch, epochs, self.best_score))
        # print detailed info
        best_info = json.dumps(self.best_results, indent=4)
        print(best_info)

        # save the best results
        print("Saving results to JSON ...")
        with open(str(self.root_dir / "val_results.json"), "w") as f:
            json.dump(self.best_results, f, indent=4)

        # results predictions
        print("Restoring best state and predicting on val set ...")
        self.model.load_state_dict(self.best_state)

        # evaluate
        _, _, predictions, _ = self.evaluate(partition="val")

        # save the predictions
        print("\n * * * Storing val set predictions  * * * \n")
        val_save_path = self.root_dir / "predictions" / "val"
        self.save_predictions(predictions=predictions, save_path=val_save_path)

    
    def _print_metrics(self, summary_str:str, metrics:dict):
        """
        Helper which prints out metrics
        """
  
        task_str = {t: summary_str for t in self.tasks}
            
        for t in self.tasks:
            task_info = self.metric.tasks_dict[t]
            m = task_info["score"]

            result = metrics[t].get(m)
            if result:
                summary_str = summary_str + " {} {}: {:.3f}, ".format(t, m, result["all"]) # update score info
                # add all other keys
                for k in result.keys():
                    if k == "all":  # not needed here
                        continue
                    task_str[t] = task_str[t] + "{} {} : {:.3f}, ".format(k, m, result[k])
            
        print(summary_str)
        for k, s in task_str.items():
            print(s)


    
    def _log_results(self, epoch, validation_score, metrics:dict):

        is_best = validation_score > self.best_score

        if is_best:
            print("* * * Found new best model with score {:05.5f} * * *".format(validation_score))
            self.best_epoch = epoch
            self.best_score = validation_score
            if metrics is not None:
                self.best_results = deepcopy(metrics)
            self.best_state = self.model.state_dict()

        # reset early stopping
        self.no_improvement = 0 

        # save best score to a dict
        entry = {self.best_epoch: self.best_score}
        self.best_scores.update(entry)

        # save to json

        # save to ckpt
        save_dict = {
            "validation_score": validation_score,
            "metrics": metrics,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
            "params": self.params
        }

        # save the checkpoint
        self.save_checkpoint(state_dict=save_dict, is_best=is_best, ckpt_path=self.ckpt_dir)

    
    def save_predictions(self, predictions:dict, save_path:Path):
        """
        Helper which converts the predictions and saves them to a csv file
        :predictions a dict returned by evaluate() Contains predictions in the form of numpy arrays for each task.
        :save_path Path to folder to save predictions csvs into
        :return None
        """
        # create folder if it does not exist
        save_path.mkdir(exist_ok=True, parents=True)

        #TODO create a pandas Dataframe

        task_info = self.metric.tasks_dict

        # create one file per task
        for task in self.tasks:
            
            data = predictions[task]

            if task_info[task]["type"] == "classification":
                # array needs to be reduced via argmax
                data = np.argmax(data, axis=1)

            col_names = []
            if task == "voc_type":
                col_names.append("Voc_Type")
            else:
                if task_info[task]["type"] == "classification":
                    col_names.extend(task_info[task]["categories"])
                else:
                    col_names.extend(task_info[task]["dimensions"])

            pred_df = pd.DataFrame(data=data, columns=col_names)

            # change order for culture to match labels file
            if task == "culture_emotion":
                surprise_cols = ["China_Surprise","United States_Surprise","South Africa_Surprise","Venezuela_Surprise"]
                reordered_cols = pred_df.columns.drop(surprise_cols).to_list() + surprise_cols
                pred_df = pred_df[reordered_cols]

            # vocal type predictions need to be mapped back to class names
            if task == "voc_type":
                pred_df["Voc_Type"] = pred_df["Voc_Type"].map(dataset.INVERSE_MAP_VOCAL_TYPES)

            # add the fids
            pred_df.insert(loc=0, column="File_ID", value=predictions["File_ID"])

            print("Created a dataframe with {} rows".format(len(pred_df)))
            # store the dataframe
            path = str(save_path / "{}.csv".format(task))
            print("Saving predictions to {}...".format(path))
            pred_df.to_csv(path, sep=",", index=False)

        print("Export of predictions complete")

    def test(self):
        """
        Testing 
        """

        # restore the best saved state
        self.model.load_state_dict(self.best_state)

        # evaluate
        _, _, predictions, _ = self.evaluate(partition="test")

        save_test_path = self.root_dir / "predictions" / "test"
        self.save_predictions(predictions=predictions, save_path=save_test_path)

    

    def evaluate(self, partition="val") -> Tuple[float, dict, dict, dict]:
        """
        Evaluation helper for validation and test
        returns a Tuple of score (for validation), metrics (for validation), predictions (for validation and test), targets (for validation)
        """

        if partition.lower() == "val":
            dl = self.val_loader
            params = self.val_params
        elif partition.lower() == "test":
            dl = self.test_loader
            params = self.test_params
        else:
            raise NotImplementedError

        # allocate
        batch_preds = {}
        batch_targets = {}
        for t in self.tasks:
            batch_preds[t] = []
            batch_targets[t] = []
        file_ids = []

        is_blocking = params.pin_memory

        self.model.eval()

        with torch.no_grad():
            # iterate loader 
            for index, batch in enumerate(dl):
                # transfer to GPU
                audio = batch["audio"].cuda(non_blocking=is_blocking)
                labels = {}

                # build a list of file ids with brackets around them
                file_ids.extend(["[{}]".format(f) for f in batch["fid"]])

                if partition.lower() != "test":

                    for t in self.tasks:
                        batch_targets[t].append(batch[t].cpu())
                        #labels.update({t: batch[t].cuda(non_blocking=is_blocking)})    # labels are not needed for fwd pass since not training here.

                outputs = self.model(audio, labels)

                for t in self.tasks:
                    batch_preds[t].append(outputs[t].cpu())

        # concatenate predictions
        pred = {}
        for t in self.tasks:
            pred[t] = torch.cat(batch_preds[t], dim=0)
        
        if partition.lower() != "test":

            # partition-wide metrics computation
            target = {}

            # concatenate targets along batch dimension
            for t in self.tasks:
                target[t] = torch.cat(batch_targets[t], dim=0)

            # compute metrics

            metrics = self.metric.compute(preds=pred, targets=target)

            # compute the scalar score 
            task_scores = []
            for i, t in enumerate(self.tasks):
                metric_to_score = self.metric.tasks_dict[t]["score"]
                result = metrics[t][metric_to_score]

                if isinstance(result, dict):
                    task_scores.append(result.get("all", 0.0))
                else:
                    task_scores.append(result)

                #if isinstance(metrics[t], np.ndarray):  # check all arrays
                #    task_scores[i] = np.mean(metrics[t])
                #else: # float
                #    task_scores[i] = metrics[t]

            score = sum(task_scores) / len(task_scores)

            # convert to numpy
            predictions = {t: pred[t].numpy() for t in pred.keys()}
            targets = {t: target[t].numpy() for t in target.keys()}

            #return score, metrics, predictions, targets

        else:  # test
            # return only the predictions
            predictions = {t: pred[t].numpy() for t in pred.keys()}
            score = -1
            metrics = {}
            targets = {}

        # add file ids
        predictions["File_ID"] = file_ids

        return score, metrics, predictions, targets

        


    def save_checkpoint(self, state_dict:dict, is_best:bool, ckpt_path:Path):
        """
        Helper which saves checkpoints with model + meta info.
        Optionally copies the checkpoint if its the best so far.
        """

        ckpt_path.mkdir(exist_ok=True, parents=True)
        filepath = ckpt_path / "last.pth.tar"
        torch.save(state_dict, str(filepath))

        if is_best:
            self.best_path = ckpt_path / "best.pth.tar"
            copy(str(filepath), str(self.best_path))

              
    def load_checkpoint(self, ckpt_path:Path):
        """
        Loads checkpoint
        """        

        return torch.load(str(ckpt_path))   


    def save_initial_checkpoint(self, state_dict:Dict):
        """
        Helper which saves the model's initial state. 
        """

        filedir = Path(self.ckpt_dir)
        if not filedir.exists():
            filedir.mkdir(parents=True, exist_ok=True)

        path = str(filedir / "initial.pth.tar")
        print("Storing initial state dict to {}".format(path))
        torch.save(state_dict, path)


    def load_initial_checkpoint(self) -> dict:
        """
        Helper which loads the model initial state
        """

        ckpt = torch.load(str(self.ckpt_dir / "initial.pth.tar"))

        return ckpt


