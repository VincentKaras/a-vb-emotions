import os
import random
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from end2you.utils import Params
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import dataset
#from dataset.data_augmentation import ChainRunner, random_pitch_shift, random_time_warp


"""
Provides datasets
Vincent Karas 08/2022
"""

def collate_fn():
    """
    specialised colllator if needed
    """
    pass


class VocalDataModule(LightningDataModule):
    """
    For PL
    """

    def __init__(self, params, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

        self.params = params

    def get_loader(self, phase:str) -> DataLoader:
        """
        Returns a dl for the partition
        :phase one of [train, val, test]
        """
        if phase == "train":
            print("Creating training dataset and loader")
            phase_params = self.params.train  # phase specific params
        elif phase == "val":
            print("Creating validation dataset and loader")
            phase_params = self.params.val
        else:
            print("Creating test dataset and loader")
            phase_params = self.params.test

        ds = VocalDataset(phase_params)

        shuffle = phase_params.is_training
        drop_last = phase_params.is_training    # only drop last batch during train

        return DataLoader(ds, phase_params.batch_size, shuffle, num_workers=8, drop_last=drop_last)

    # override methods
    def train_dataloader(self):
        return self.get_loader("train")
    def val_dataloader(self):
        return self.get_loader("val")
    def test_dataloader(self):
        return self.get_loader("test")


class VocalDataset(Dataset):
    """
    Loads the complete data for a partition via csv file into pandas - important: All info is included for the labels. So batch includes:
    - Raw audio data
    - Possibly feature data
    - Country of Subject
    - High
    - Two [Valence, Arousal]
    - China Emotions 
    - United States Emotions
    - South Africa Emotions
    - Venezuela Emotions
    For test files, all the label fields will be empty
    """

    def __init__(self, params:Params) -> None:
        super().__init__()
        self.params = params
        
        # set paths
        self.wav_folder = Path(params.wav_folder)
        self.dataset_file = Path(params.dataset_file)
        if not self.wav_folder.exists():
            print("Specified audio folder {} does not exist! Falling back to default {}...".format(str(self.wav_folder), str(dataset.DATA_DIR)))
            self.wav_folder = dataset.DATA_DIR
        if not self.dataset_file.exists():
            if str(self.params.partition).lower() == "train":
                default = dataset.TRAIN_FILE
            elif str(self.params.partition).lower() == "val":
                default = dataset.VAL_FILE
            else:
                default = dataset.TEST_FILE
            print("Specified label file {} does not exist! Falling back to default {} ...".format(str(self.dataset_file), str(default)))
            self.dataset_file = default
        
        self.sr = params.sr
        self.max_wav_length = params.window_size

        # switch to train/val/test
        self.partition = params.partition

        # load the csv
        #csv_path = Path(self.label_path) / ("{}.csv".format(self.process))

        self.meta = pd.read_csv(str(self.dataset_file), header="infer", dtype={"File_ID": "str"})

        #self.type_map = {t: i for i, t in enumerate(dataset.VOCAL_TYPES)}    # maps the string categories to numbers 0-7
        #self.country_map = {c: i for i, c in enumerate(dataset.CULTURES)}
        self.type_map = dataset.MAP_VOCAL_TYPES
        self.country_map = dataset.MAP_CULTURES

        # data augmentation enable/disable
        self.augment = params.augment
        if self.augment:
            pass    # data augmentation is now done as specaugment directly in the SSL model 
            """
            chain = augment.EffectChain()
            pitch = params.augment.pitch
            warp = params.augment.time_warp
            chain.pitch(partial(random_pitch_shift, a=0-pitch, b=pitch)).rate(self.sr)
            chain.tempo(partial(random_time_warp, f=warp))
            self.chain = ChainRunner(chain)
            """

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        out = {}

        wav_id = self.meta.loc[index, "File_ID"]
        out["fid"] = wav_id
        # load audio
        wav = self.load_wav(wav_id)
        out["audio"] = wav
        # labels

        if str(self.params.partition).lower() != "test":

            # categorical
            out["country"] = self.country_map[self.meta.loc[index, "Country"]]
            out["voc_type"] = self.type_map[self.meta.loc[index, "Voc_Type"]]
            # continuous
            # low 
            low = self.meta.iloc[index, 3:5].to_numpy("float32") # valence, arousal
            out["low"] = low
            # high
            high = self.meta.iloc[index, 5:15].to_numpy("float32")   # 10 categorical emotions in fixed order
            out["high"] = high
            # culture specific emotions
            """
            emotion_china = self.meta.iloc[index, 15:25].to_numpy("float32")
            emotion_us = self.meta.iloc[index, 25:35].to_numpy("float32")
            emotion_south_africa = self.meta.iloc[index, 35:45].to_numpy("float32")
            emotion_venezuela = self.meta.iloc[index, 45:55].to_numpy("float32")
            out["emotion_china"] = emotion_china
            out["emotion_us"] = emotion_us
            out["emotion_south_africa"] = emotion_south_africa
            out["emotion_venezuela"] = emotion_venezuela
            """
            # bundle all 40 (4x10) culture emotions
            culture_emotion = self.meta.iloc[index, 15:55].to_numpy("float32")
            out["culture_emotion"] = culture_emotion

        else:   # test has no labels, return dict without keys or with none in them
            pass

        return out


    def load_wav(self, id):
        """
        Loads a single wav file
        """

        wav, sr = torchaudio.load(str(self.wav_folder / "{}.wav".format(id)))
        # remove stereo
        if wav.shape[0] > 1:
            wav = wav[0, :]
            wav = torch.unsqueeze(0)

        assert sr == self.sr, "Sample rate of audio does not match"

        # data augmentation for training
        #if self.process == "train" and self.augment:
        #    wav = self.chain(wav)   # run through the chain
            

        max_length = int(self.sr * self.max_wav_length)
        # replicate

        # truncate 
        if wav.shape[1] > max_length:
            if self.augment:
                start = random.randint(0, wav.shape[-1] - max_length)
                wav = wav[:, start:start + max_length]
            else:
                wav = wav[:, :max_length]
        
        # pad with zeros to the right
        elif wav.shape[1] < max_length:
            wav = F.pad(wav, [0, max_length - wav.shape[1]])

        # squeeze wav
        wav = torch.squeeze(wav)

        return wav


    def set_node(self, path:str) -> str:
        """
        Helper which sets the wav_folder and dataset_file to files on the proper slurm node 
        """

        return ""
        


