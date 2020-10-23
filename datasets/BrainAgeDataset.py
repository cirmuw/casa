from torch.utils.data.dataset import Dataset
import pandas as pd
from py_jotools import augmentation, mut
import numpy as np
import nibabel as nib
import torch

class BrainAgeDataset(Dataset):

    def __init__(self, datasetfile, split=['base_train'], iterations=None, batch_size=None, res=None):

        df = pd.read_csv(datasetfile, index_col=0)
        if type(split) is list:
            selection = np.any([df.split==x for x in split], axis=0)
        else:
            selection = df.split==split

        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if res is not None:
            self.df = self.df.loc[self.df.Scanner==res]
            #self.df = self.df.reset_index()

        if iterations is not None:
            self.df = self.df.sample(iterations*batch_size, replace=True)
            self.df = self.df.reset_index(drop=True)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        nimg = nib.load(self.df.iloc[index].Image)
        nimg = nib.as_closest_canonical(nimg)
        img = nimg.get_fdata()
        img = img.swapaxes(0, 2)
        img = mut.resize(img, (64, 128, 128))
        img = mut.norm01(img)
        img = img[None, :, :, :]

        return torch.tensor(img).float(), torch.tensor(self.df.iloc[index].Age).float(), self.df.iloc[index].Image, self.df.iloc[index].Scanner
