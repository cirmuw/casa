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
import random

class MDTLUNADataset(Dataset):

    def __init__(self, datasetfile, split=['train'], iterations=None,
                 batch_size=None, res=None, labelDebug=None, n_slices=1, cropped_to=(288, 288)):
        df = pd.read_csv(datasetfile, index_col=0)
        if type(split) is list:
            selection = np.any([df.split==x for x in split], axis=0)
        else:
            selection = df.split==split

        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if res is not None:
            self.df = self.df.loc[self.df.res==res]
            #self.df = self.df.reset_index()

        if labelDebug is not None:
            self.df = self.df.loc[self.df.label==labelDebug]

        if iterations is not None:
            self.df = self.df.sample(iterations*batch_size, replace=True)
            self.df = self.df.reset_index(drop=True)

        self.n_slices = n_slices

        self.cropped_to = cropped_to


    def __len__(self):
        return len(self.df)


    def load_image(self, path, channels=1, shiftx_aug=0, shifty_aug=0):
        if self.n_slices==1:
            img = pyd.read_file(path).pixel_array

            if self.cropped_to is not None:
                w = img.shape[0]
                s1 = int((w-self.cropped_to[0])/2)
                e1 = int(s1 + self.cropped_to[0])

                h = img.shape[1]
                s2 = int((h - self.cropped_to[1]) / 2)
                e2 = int(s2 + self.cropped_to[1])
                img = img[s1 + shiftx_aug:e1 + shiftx_aug, s2 + shifty_aug:e2 + shifty_aug]

            img = mut.intensity_window(img, low=-1024, high=800)
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


    def load_annotation(self, elem, shiftx_aug=0, shifty_aug=0):
        dcm = pyd.read_file(elem.image)

        x = elem.coordX
        y = elem.coordY

        if self.cropped_to is not None:
            x -= (dcm.Rows-self.cropped_to[0])/2
            y -= (dcm.Columns-self.cropped_to[1])/2


        x -= shifty_aug
        y -= shiftx_aug

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

        shiftx_aug = random.randint(-50, 50)
        shifty_aug = random.randint(-50, 50)

        img = self.load_image(elem.image, shiftx_aug=shiftx_aug, shifty_aug=shifty_aug)


        batch = dict()
        batch['data'] = img
        batch['roi_labels'] = np.array([elem.bin_malignancy])
        batch['bb_target'] = self.load_annotation(elem, shiftx_aug=shiftx_aug, shifty_aug=shifty_aug)
        batch['scanner'] =  elem.res
        batch['img'] = elem.image

        return batch
