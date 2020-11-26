from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import pandas as pd
from py_jotools import augmentation, mut
import numpy as np
import nibabel as nib
import torch
import pydicom as pyd
import math


class MDTLUNADatasetContinuous(Dataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['ges', 'geb', 'sie'], n_slices=1, seed=None):
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

        self.n_slices = n_slices


    def __len__(self):
        return len(self.df)


    def load_image(self, path, channels=1):
        if self.n_slices==1:
            img = pyd.read_file(path).pixel_array
            img = mut.intensity_window(img, low=-1024, high=400)
            img = mut.norm01(img)


            if channels==3:
                return np.tile(img, [3, 1, 1])
            else:
                return img[None, :, :]
        else:
            reader = sitk.ImageSeriesReader()

            dicom_names = reader.GetGDCMSeriesFileNames(os.path.dirname(path))
            idx_slice = dicom_names.index(path)
            dicom_names = dicom_names[idx_slice - math.floor(self.n_slices/2): idx_slice + math.ceil(self.n_slices/2)]
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            img = sitk.GetArrayFromImage(image)
            img = mut.intensity_window(img, low=-1024, high=400)
            img = mut.norm01(img)

            return img


    def load_annotation(self, elem):
        dcm = pyd.read_file(elem.image)
        x = elem.coordX
        y = elem.coordY

        diameter = elem.diameter_mm
        spacing = float(dcm.PixelSpacing[0])

        x -= int((diameter / spacing) / 2)
        y -= int((diameter / spacing) / 2)

        x2 = x+int(diameter/spacing)
        y2 = y+int(diameter/spacing)

        box = np.zeros((1, 4))
        box[0, 0] = y
        box[0, 1] = x
        box[0, 2] = y2
        box[0, 3] = x2

        return box

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img = self.load_image(elem.image)

        batch = dict()
        batch['data'] = img
        batch['roi_labels'] = np.array([elem.label])
        batch['bb_target'] = self.load_annotation(elem)
        batch['scanner'] =  elem.res
        batch['img'] = elem.image

        return batch
