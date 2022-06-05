import math
import os
import ast
import librosa

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
    ):
        ignore_labels = ('akiapo', 'aniani', 'apapan', 'barpet', 'crehon', 'elepai', 'ercfra',
                         'hawama', 'hawcre', 'hawgoo', 'hawhaw', 'hawpet1', 'houfin', 'iiwi',
                         'jabwar', 'maupar', 'omao', 'puaioh', 'skylar', 'warwhe1', 'yefcan')
        df2021 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2021/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2022 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2022/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2021["data_year"] = 2021
        df2022["data_year"] = 2022
        df = pd.concat([df2021, df2022], ignore_index=True)
        print(len(df))
        df = df[~df.primary_label.isin(ignore_labels)]
        print(len(df))
        labels = list(set(df.primary_label.unique()))
        labels.sort()
        self.labels = labels
        self.df = df
        self.dataset_dir = dataset_dir


        self.duration = 15
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.bird2id = {x: idx for idx, x in enumerate(labels)}

    def load_one(self, filename, offset, duration):
        try:
            wav, _ = librosa.load(filename, sr=None, offset=offset, duration=duration)
        except:
            print("failed reading", filename)
        return wav


    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year == 2022 else "train_short_audio", row["primary_label"], row['filename'].split("/")[-1])

        ## wav

        wav_len_sec = librosa.get_duration(filename=filename, sr=None)
        duration = self.duration
        max_offset = wav_len_sec - duration
        max_offset = max(max_offset, 1)
        offset = np.random.randint(max_offset)


        wav = self.load_one(filename, offset=offset, duration=self.duration)
        if wav.shape[0] < (self.dsr):
            wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        ## labels
        labels = torch.zeros((len(self.labels),))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 1.0
            except:
                continue

        return {
            "wav": torch.tensor(wav).unsqueeze(0),
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)
