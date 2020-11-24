import models.MDTRetinaNet as mdtr
import logging
import torch
from datasets.MDTLUNADataset import MDTLUNADataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

def train_loop(model, dl, optimizer, savepath, scheduler=None, epochs=50):
    for epoch in range(epochs):
        running_loss = 0.0
        running_class_loss = 0.0
        running_box_loss = 0.0
        for iter_num, data in enumerate(dl):
            optimizer.zero_grad()

            results_dict = model.train_forward(data)

            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            running_loss += results_dict['monitor_values']['loss']
            running_class_loss += results_dict['monitor_values']['class_loss']
            running_box_loss += results_dict['monitor_values']['box_loss']

        print(epoch, 'loss:', running_loss / len(dl), 'class:', running_class_loss / len(dl), 'box:',
              running_box_loss / len(dl))
        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), savepath)