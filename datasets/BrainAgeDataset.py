from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from active_catinous.utils import *

class BrainAgeDataset(Dataset):

    def __init__(self, datasetfile, split=['base_train'], iterations=None, batch_size=None, res=None, seed=None, balance=True):

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
            if balance:
                self.df = self.df.sample(iterations*batch_size, replace=True, random_state=seed, weights='weight')
            else:
                self.df = self.df.sample(iterations*batch_size, replace=True, random_state=seed)
            self.df = self.df.reset_index(drop=True)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        nimg = nib.load(self.df.iloc[index].Image)
        nimg = nib.as_closest_canonical(nimg)
        img = nimg.get_fdata()
        img = img.swapaxes(0, 2)
        img = resize(img, (64, 128, 128))
        img = norm01(img)
        img = img[None, :, :, :]

        return torch.tensor(img).float(), torch.tensor(self.df.iloc[index].Age).float(), self.df.iloc[index].Image, self.df.iloc[index].Scanner
