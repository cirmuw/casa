from torch.utils.data.dataset import Dataset
import os
import pandas as pd
from py_jotools import augmentation, mut
import numpy as np
import nibabel as nib
import torch
import pydicom as pyd
import random


class LIDCDatasetContinuous(Dataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['ges', 'geb', 'sie'], seed=None):
        df = pd.read_csv(datasetfile, index_col=0)
        assert (set(['train']).issubset(df.split.unique()))

        np.random.seed(seed)

        res_dfs = list()
        for r in order:
            res_df = df.loc[df.res == r]
            res_df = res_df.loc[res_df.split == 'train']
            res_df = res_df.sample(frac=1, random_state=seed)

            res_dfs.append(res_df.reset_index(drop=True))

        combds = None
        new_idx = 0

        for j in range(len(res_dfs) - 1):
            old = res_dfs[j]
            new = res_dfs[j + 1]

            old_end = int((len(old) - new_idx) * transition_phase_after) + new_idx
            if combds is None:
                combds = old.iloc[:old_end]
            else:
                combds = combds.append(old.iloc[new_idx + 1:old_end])

            old_idx = old_end
            old_max = len(old) - 1
            new_idx = 0
            i = 0

            while old_idx <= old_max and (i / ((old_max - old_end) * 2) < 1):
                take_newclass = np.random.binomial(1, min(i / ((old_max - old_end) * 2), 1))
                if take_newclass:
                    combds = combds.append(new.iloc[new_idx])
                    new_idx += 1
                else:
                    combds = combds.append(old.iloc[old_idx])
                    old_idx += 1
                i += 1
            combds = combds.append(old.iloc[old_idx:])

        combds = combds.append(new.iloc[new_idx:])
        combds.reset_index(inplace=True, drop=True)
        self.df = combds

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        img = pyd.read_file(path).pixel_array
        img = mut.intensity_window(img, low=-1024, high=1500)
        img = mut.norm01(img)

        return np.tile(img, [3, 1, 1])

    def load_annotation(self, elem):
        dcm = pyd.read_file(elem.image)
        x = elem.coordX
        y = elem.coordY
        diameter = elem.diameter_mm
        spacing = float(dcm.PixelSpacing[0])

        x -= int((diameter / spacing) / 2)
        y -= int((diameter / spacing) / 2)

        x2 = x + int(diameter / spacing)
        y2 = y + int(diameter / spacing)

        box = np.zeros((1, 4))
        box[0, 0] = x
        box[0, 1] = y
        box[0, 2] = x2
        box[0, 3] = y2

        return box

    def __getitem__(self, index):
        elem = self.df.iloc[index]

        img = self.load_image(elem.image)
        annotation = self.load_annotation(elem)

        target = {}
        target['boxes'] = torch.as_tensor(annotation, dtype=torch.float32)
        target['labels'] = torch.as_tensor((elem.bin_malignancy + 1,), dtype=torch.int64)
        target['image_id'] = torch.tensor([index])
        target['area'] = torch.as_tensor(
            ((annotation[:, 3] - annotation[:, 1]) * (annotation[:, 2] - annotation[:, 0])))
        target['iscrowd'] = torch.zeros((1,), dtype=torch.int64)

        return torch.as_tensor(img, dtype=torch.float32), target, elem.res, elem.image