import math
import os
import ast
import random
import traceback
from builtins import Exception

import librosa

import numpy as np
import pandas as pd
import audiomentations as AA

import torch
from torch.utils.data import Dataset

train_aug = AA.Compose(
    [
        AA.AddBackgroundNoise(
            sounds_path="input/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/aicrowd2020_noise_30sec/noise_30sec", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
    ]
)


class BirdDataset(Dataset):
    def __init__(
            self,
            mode: str,
            folds_csv: str,
            dataset_dir: str,
            fold: int = 0,
            n_classes: int = 21,
            transforms=None,
            multiplier: int = 1,
            duration: int = 30,
            val_duration: int = 5,
    ):
        ## many parts from https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/main/data/ps_ds_2.py
        self.folds_csv = folds_csv
        self.df = pd.read_csv(folds_csv)
        # take sorted labels from full df
        birds = sorted(list(set(self.df.primary_label.values)))
        print('Number of primary labels ', len(birds))
        if mode =="train":
            self.df = self.df[self.df['fold'] != fold]
        else:
            self.df = self.df[self.df['fold'] == fold]

        self.dataset_dir = dataset_dir

        self.mode = mode

        self.duration = duration if mode == "train" else val_duration
        self.sr = 32000
        self.dsr = self.duration * self.sr

        self.n_classes = n_classes
        self.transforms = transforms

        self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)
        vc = self.df.primary_label.value_counts()
        dataset_length = len(self.df)
        label_weight = {}
        for row in vc.items():
            label, count = row
            label_weight[label] = math.pow(dataset_length / count, 1 / 2)

        self.df["label_weight"] = self.df.primary_label.apply(lambda x: label_weight[x])

        self.bird2id = {x: idx for idx, x in enumerate(birds)}

        ## TODO: move augmentation assignment outside of dataset
        if self.mode == "train":
            print(f"mode {self.mode} - augmentation is active {train_aug}")
            self.transforms = train_aug
            if multiplier > 1:
                self.df = pd.concat([self.df] * multiplier, ignore_index=True)

    def load_one(self, filename, offset, duration):
        #try:
        wav, sr = librosa.load(filename, sr=None, offset=offset, duration=duration)
        
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        #except:
            #print("failed reading", filename)
        return wav

    def get_weights(self):
        return self.df.label_weight.values

    def __getitem__(self, i):
        tries = 0
        while tries < 20:
            try:
                tries += 1
                return self.getitem(i)
            except:
                traceback.print_stack()
                return self.getitem(random.randint(0, len(self) - 1))
        raise Exception("OOPS, something is wrong!!!")

    def getitem(self, i):
        row = self.df.iloc[i]
        if 'pretrain' in self.folds_csv:
            filename = os.path.join(self.dataset_dir, f"{row['filename'].split('.')[0]}.ogg")
            if not os.path.exists(filename):
                filename = filename.replace(".ogg", ".wav")
        else:
            if 'only_ml' in self.folds_csv:
                filename = os.path.join(self.dataset_dir, 'shared', row['filename'])
            elif 'pseudo' in self.folds_csv:
                filename = os.path.join(self.dataset_dir, 'shared', row['filename'])
            else:
                data_year = row['data_year']
                filename = os.path.join(self.dataset_dir, f"birdclef-{int(data_year)}",
                                        "train_audio" if data_year == 2022 else "train_short_audio", row['filename'])

        ## wav
        if self.mode == "train":
            wav_len_sec =  librosa.get_duration(filename=filename)
            duration = self.duration
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        if self.mode == "val": offset = 0

        wav = self.load_one(filename, offset=offset, duration=self.duration)
        if wav.shape[0] < (self.dsr): wav = np.pad(wav, (0, self.dsr - wav.shape[0]))
        if self.transforms: wav = self.transforms(wav, self.sr)

        ## labels
        labels = torch.zeros((self.n_classes,))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 1.0
            except:
                ## if not in 21 classes, ignore
                continue

        ## weight
        weight = torch.tensor(row['weight'])

        return {
            "wav": torch.tensor(wav).unsqueeze(0),
            "labels": labels,
            "weight": weight
        }

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    print("run")
    dataset = BirdDataset(mode="train", folds_csv="../pseudo_set_1.csv", dataset_dir="/mnt/d/kaggle/input/", fold=0, transforms=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    for batch in dataloader: break
    print(batch['wav'].shape)
