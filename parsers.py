"""
CLI args file
Vincent Karas 08/2022
"""

from argparse import ArgumentParser
import argparse
from ast import parse
from pathlib import Path
from end2you.utils import Params
from typing import Dict
import json
from datetime import datetime

class Options():
    """
    Helper class wrapping around parser
    """

    def __init__(self) -> None:
        self._parser = self._add_parsers()
        self._is_parsed = False
        self._is_initialised = False
        self._is_training = False
        self._process = ""

        self._params = None
        self._date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._dict_options = {}

    def _add_train_args(self, parser:ArgumentParser) -> ArgumentParser:
        """
        Internal helper to add train args
        """
        # dataset
        parser.add_argument("--num_train_workers", type=int, default=4,
                            help="Number of workers for loading train data (defaults to 4")
        parser.add_argument("--pin_memory", type=bool, default=False, help="Whether to use pinned memory for the dataloaders. May result in issues! Default False")
        parser.add_argument("--train_dataset_file", type=str, required=True, help="Path to the csv file that stores the labels")
        parser.add_argument("--wav_folder", type=str, help="Path to the folder where the audio files are stored")
        #parser.add_argument("--augment_pitch", type=int, default=200, help="Pitch shift augmentation")
        #parser.add_argument("--augment_time_warp", type=float, default=1.0, help="Time warping (default 1.0)")
        # optimiser
        parser.add_argument("--optimizer", type=str, default="Adam", choices=["sgd", "rmsprop", "adam", "adamw"],
                            help="Type of optimizer to use for training (defaults to Adam)")
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
        parser.add_argument("--fe_lr", type=float, default=1e-5, help="Learning rate for feature extractor (default 1e-5)")
        # training
        parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
        parser.add_argument("--num_epochs", type=int, help="Number of epochs to train")
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
        parser.add_argument("--loss", type=str, default="ccc", help="Loss to use for the regression tasks. Default is CCC loss")
        parser.add_argument("--loss_strategy", type=str, default="mean", choices=["mean", "rruw", "druw", "dwa"], help="Strategy to balance the loss weighting of multiple tasks. Can use simple mean, dynamic weight average or uncertainty (default mean)")

        # model weights
        parser.add_argument("--pretrained_path", type=str, default=None, help="Path to checkpoint file of pretrained model")

        # TODO loss weights, weighting scheme

        # logging
        parser.add_argument("--save_summary_steps", type=int, default=500,
                            help="Perform evaluation every n steps during training")

        return parser


    def _add_eval_args(self, parser:ArgumentParser) -> ArgumentParser:
        """
        Interal helper to add evaluation (val and test) args
        """
        # val 
        parser.add_argument("--num_val_workers", type=int, default=4,
                            help="Number of workers for loading eval data (defaults to 4")
        parser.add_argument("--val_dataset_file", type=str, required=True, help="Path to the csv file that stores the validation labels")
        # test 
        parser.add_argument("--num_test_workers", type=int, default=4, help="Number of workers for loading test data (default 4)")
        parser.add_argument("--test_dataset_file", type=str, required=True, help="Path to the csv file that holds test file info")
       
        parser.add_argument("--metrics", type=str, help="Metric to use for validation", default="ccc")  # TODO maybe do this via config file per task?
        parser.add_argument("--prediction_file", type=str, default="predictions.csv",
                            help="The file to write test predictions in csv format")

        return parser


    def _add_test_args(self, parser:ArgumentParser) -> ArgumentParser:
        """
        Internal helper to add test args
        """
        parser.add_argument("--num_test_workers", type=int, default=4, help="Number of workers for loading test data (default 4)")
        parser.add_argument("--test_dataset_file", type=str, required=True, help="Path to the csv file that holds test file info")
        parser.add_argument("--wav_folder", type=str, help="Path to the folder where the audio files are stored")   # redundant? or not if only one is loaded?
        parser.add_argument("--model_path", type=str, required=True,
                            help="Path to the model to test")
        parser.add_argument("--metrics", type=str, help="Metric to use for testing", default="ccc")  # TODO maybe do this via config file per task?
        parser.add_argument("--prediction_file", type=str, default="predictions.csv",
                            help="The file to write predictions in csv format") # TODO do for different tasks?

    
    def _add_parsers(self) -> argparse.ArgumentParser():
        """
        construct the parser objects
        Adds arguments that are always needed, then adds process-specific arguments via subparsers
        :return: An ArgumentParser
        """

        # create the parser
        parser = argparse.ArgumentParser(description="A parser for this cool project. Adds flags")   


        parser.add_argument("--name", type=str, default="experiment_1", help="Experiment ID")
        # MODEL
        parser.add_argument("--model_name", type=str, default="basemtl", choices=["basemtl", "stacked", "stacked_v2", "attnfusion", "attnbranch", "attnbranch_v2"], help="Name of model architecture to use")
        parser.add_argument("--feature_extractor", type=str, default="wav2vec2-base", choices=["wav2vec2-base"], help="Feature network architecture to use")
        parser.add_argument("--num_outputs", type=int, default=4, help="Number of model outputs (default 4)")
        parser.add_argument("--num_tasks", type=int, default=4, help="Number of tasks to train/predict for. Default is 4")
        parser.add_argument("--task", type=str, default="all", choices=["all"], help="Task to train/evaluate. Currently has no effect (default all).")
        parser.add_argument("--features", type=str, default="attention", choices=["cnn", "attention", "both"], help="Type of feature to extract from the feature extractor. CNN embedding, last layer attention, or both")
        parser.add_argument("--embedding_size", type=int, default=256, help="Dimensionality of hidden layer embeddings. Default 256")
        parser.add_argument("--activation", type=str, choices=["gelu"], default="gelu", help="Activation function to use. Default GELU")
        parser.add_argument("--pool", type=str, default="attention", choices=["avg", "max", "attention"], help="Type of temporal pooling to use (dafault attention)")
        parser.add_argument("--branch_layers", type=str, default="even", choices=["first", "middle", "last", "even"], help="Transformer layer hidden states to use as inputs for the branching model. Default is even, which means [3,6,9,12]")
        parser.add_argument("--chain_order", type=str, default="fixed", choices=["fixed", "perf"], help="Order to chain emotions for stacked_v2 arch. fixed (standard order) or perf (descending baseline performance). Defaults to fixed")
       
        # Data
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default 8)")
        parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate in Hz. (defaults to 16000)")
        parser.add_argument("--window_size", type=float, default=2.5, help="Length of an audio clip in seconds (default 2.5)")

        #SpecAugment
        parser.add_argument("--mask_time_prob", type=float, default=0.05, help="SpecAugment timestep mask probability (default 0.05)")
        parser.add_argument("--mask_time_length", type=int, default=10, help="SpecAugment time mask length (default 10)")
        parser.add_argument("--mask_feature_prob", type=float, default=0.0, help="SpecAugment feature mask probability (default 0.0)")
        parser.add_argument("--mask_feature_length", type=int, default=10, help="SpecAugment feature mask length (default 10)")

        # GPU
        parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA. Defaults to True")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (defaults to 1)")


        # Path
        parser.add_argument("--root_dir", type=str, default="./embed", help="Root folder for the output")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Models are saved here")
        parser.add_argument("--log_dir", type=str, default="./logs", help="Logs are saved here")

        # TODO
        # 
        #   
        #       

        subparsers = parser.add_subparsers(help="Should be [train, test]", required=True, dest="process")

        # add to the parser

        # training
        train_subparser = subparsers.add_parser(name="train", help="Training arguments")
        train_subparser = self._add_train_args(train_subparser)
        train_subparser = self._add_eval_args(train_subparser)
        # testing
        test_subparser = subparsers.add_parser(name="test", help="Testing arguments")
        test_subparser = self._add_test_args(test_subparser)

        self._is_initialised = True

        return parser

    def parse(self) -> Params:
        """
        Helper that parses the cli and stores information
        :return Params object
        """

        if not self._is_initialised:
            self._add_parsers()

        args = self._parser.parse_args()
        self._dict_options = vars(args)
        self._process = args.process

        # read out 
        if "train" in self._process.lower():
            self._params = self._get_train_params()
        elif "test" in self._process.lower():
            self._params = self._get_test_params()
        else:
            raise ValueError("Operation {} not recognized".format(self._process))

        print("Current date: ", self._date)
        # print out args
        self._print(self._dict_options)
        # save args to json
        self._save(self._dict_options)

        self._is_parsed = True  # flag to only execute this once

        return self._params


    def _print(self, args:Dict):
        """
        Helper that prints the CLI args out as a sorted list
        """
        print("-" * 20 + " Options " + "-" * 20)
        for k, v in sorted(args.items()):
            print("{} : {}".format(str(k), str(v)))
        print("-" * 49)


    def _save(self, args:Dict):
        """
        Helper function that saves the commandline arguments to a file in JSON format
        :param args: A dictionary created from the commandline args
        :return:
        """

        #experiment_dir = Path(self._params.checkpoints_dir) / self._params.name
        experiment_dir = Path(self._params.checkpoints_dir)  / "options"    
        #print(experiment_dir)
        #if self._is_training and not experiment_dir.exists():
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            assert experiment_dir.exists(), "Experiment dir {} does not exist".format(experiment_dir)

        # save as json file
        file_name = experiment_dir / "options_{}.json".format(self._process)
        with open(file_name, "w") as f:
            json.dump(args, f, indent=6)


    def _get_train_params(self) -> Params:
        """
        Helper which assembles a Params object for the train process
        """
        
        # check the directories, if necessary, create new run
        root_path = Path(self._dict_options["root_dir"])
        if root_path.exists():
            #root_dir = self._dict_options["root_dir"]
            print("Warning: Root folder {} already exists!".format(str(root_path)))
            print("Creating new subfolder with current date and changing logs and ckpt dirs to that folders subfolders...")
            self._dict_options["root_dir"] = str(root_path / str(self._date))
            self._dict_options["checkpoints_dir"] = str(root_path / str(self._date) / "checkpoints")
            self._dict_options["log_dir"] = str(root_path / str(self._date) / "logs")

        train_params = Params(dict_params={
            "process": "train",
            "train": Params(dict_params={
                "loss": self._dict_options["loss"],
                "loss_strategy": self._dict_options["loss_strategy"],
                "dataset_file": self._dict_options["train_dataset_file"],
                "wav_folder": self._dict_options["wav_folder"],
                "optimizer": self._dict_options["optimizer"],
                "lr": self._dict_options["lr"],
                "fe_lr": self._dict_options["fe_lr"],
                "weight_decay": self._dict_options["weight_decay"],
                "num_epochs": self._dict_options["num_epochs"],
                "seed": self._dict_options["seed"],
                "cuda": self._dict_options["cuda"],
                "is_training": True,
                "partition": "Train",
                "sr": self._dict_options["sr"],
                "window_size": self._dict_options["window_size"],
                "batch_size": self._dict_options["batch_size"],
                "num_workers": self._dict_options["num_train_workers"],
                "pin_memory": self._dict_options["pin_memory"],
                "save_summary_steps": self._dict_options["save_summary_steps"],
                "augment": Params(dict_params={
                    "mask_time_prob":  self._dict_options["mask_time_prob"],
                    "mask_time_length": self._dict_options["mask_time_length"],
                    "mask_feature_prob": self._dict_options["mask_feature_prob"],
                    "mask_feature_length": self._dict_options["mask_feature_length"],
                    }),
                #"augment": Params(dict_params={
                #    "pitch": self._dict_options["augment_pitch"],
                #    "time_warp": self._dict_options["augment_time_warp"],
            }),
            # validation params
            "val": Params(dict_params={
                "dataset_file": self._dict_options["val_dataset_file"],
                "wav_folder": self._dict_options["wav_folder"],
                "cuda": self._dict_options["cuda"],
                "is_training": False,
                "partition": "Val",
                "sr": self._dict_options["sr"],
                "window_size": self._dict_options["window_size"],
                "batch_size": self._dict_options["batch_size"],
                "num_workers": self._dict_options["num_val_workers"],
                "pin_memory": self._dict_options["pin_memory"],
                "save_summary_steps": self._dict_options["save_summary_steps"],
                "augment": False,
                "metrics": self._dict_options["metrics"]
            }),
            # train process includes a test step at the end for convenience
            "test": Params(dict_params={
                "dataset_file": self._dict_options["test_dataset_file"],
                "wav_folder": self._dict_options["wav_folder"],
                "cuda": self._dict_options["cuda"],
                "num_workers": self._dict_options["num_test_workers"],
                "pin_memory": self._dict_options["pin_memory"],
                "is_training": False,
                "partition": "Test",
                "augment": False,
                "sr": self._dict_options["sr"],
                "window_size": self._dict_options["window_size"],
                "batch_size": self._dict_options["batch_size"],
                "metrics": self._dict_options["metrics"],
            }),

            "model": Params(dict_params={
                "num_outputs": self._dict_options["num_outputs"],
                "model_name": self._dict_options["model_name"],
                "pretrained_path": self._dict_options["pretrained_path"],
                "feature_extractor": self._dict_options["feature_extractor"],
                "features": self._dict_options["features"],
                "embedding_size": self._dict_options["embedding_size"],
                "activation": self._dict_options["activation"],
                "pool": self._dict_options["pool"],
                "branch_layers": self._dict_options["branch_layers"],
                "chain_order": self._dict_options["chain_order"],
                "augment": Params(dict_params={
                    "mask_time_prob":  self._dict_options["mask_time_prob"],
                    "mask_time_length": self._dict_options["mask_time_length"],
                    "mask_feature_prob": self._dict_options["mask_feature_prob"],
                    "mask_feature_length": self._dict_options["mask_feature_length"],
                    }),
                
            }),

            "name": self._dict_options["name"],
            "root_dir": self._dict_options["root_dir"],
            "checkpoints_dir": self._dict_options["checkpoints_dir"],
            "log_dir": self._dict_options["log_dir"],
            "prediction_file": self._dict_options["prediction_file"],
            "cuda": self._dict_options["cuda"],
            "num_gpus": self._dict_options["num_gpus"],
            "num_tasks": self._dict_options["num_tasks"],
            "task": self._dict_options["task"],

        })

        return train_params

    
    def _get_test_params(self) -> Params:
        """
        Helper which assembles a Params object for the test process
        """
        
        test_params = Params(dict_params={
            "process": "test",
            "name": self._dict_options["name"],
            "prediction_file": self._dict_options["prediction_file"],
            "test": Params(dict_params={
                "dataset_path": self._dict_options["test_dataset_file"],
                "cuda": self._dict_options["cuda"],
                "num_workers": self._dict_options["num_test_workers"],
                "pin_memory": self._dict_options["pin_memory"],
                "is_training": False,
                "partition": "Test",
                "augment": False,
                "sr": self._dict_options["sr"],
                "wav_folder": self._dict_options["wav_folder"],
                "batch_size": self._dict_options["batch_size"],
                "metrics": self._dict_options["metrics"],
            }),

            "model": Params(dict_params={
                "num_outputs": self._dict_options["num_outputs"],
                "model_name": self._dict_options["model_name"],
                "feature_extractor": self._dict_options["feature_extractor"],
                "features": self._dict_options["features"],
                "embedding_size": self._dict_options["embedding_size"],
                "activation": self._dict_options["activation"],
                "pool": self._dict_options["pool"],
                "branch_layers": self._dict_options["branch_layers"],
                "chain_order": self._dict_options["chain_order"],
                "augment": Params(dict_params={
                    "mask_time_prob":  self._dict_options["mask_time_prob"],
                    "mask_time_length": self._dict_options["mask_time_length"],
                    "mask_feature_prob": self._dict_options["mask_feature_prob"],
                    "mask_feature_length": self._dict_options["mask_feature_length"],
                    }),
            }),
            
            "root_dir": self._dict_options["root_dir"],
            "checkpoints_dir": self._dict_options["checkpoints_dir"],
            "log_dir": self._dict_options["log_dir"],
            "cuda": self._dict_options["cuda"],
            "num_gpus": self._dict_options["num_gpus"],
            "num_tasks": self._dict_options["num_tasks"],
            "task": self._dict_options["task"],

        })

        return test_params