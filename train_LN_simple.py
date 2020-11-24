import models.MDTRetinaNet as mdtr
import logging
import torch
from datasets.MDTLUNADataset import MDTLUNADataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import models.MDTRetinaNet as mdtr


def train_loop(ds_path, ds_split, savepath, lr=1e-4, scheduler_steps=10, n_slices=1, epochs=50):
    device = torch.device('cuda')
    cf = mdtr.config(n_slices=n_slices)
    logger = logging.getLogger('medicaldetectiontoolkit')
    logger.setLevel(logging.DEBUG)
    model = mdtr.net(cf, logger)

    model.to(device)

    ds = MDTLUNADataset(ds_path, split=ds_split, n_slices=n_slices)
    dl = DataLoader(ds, batch_size=4, num_workers=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler_steps is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_steps, gamma=0.1)
    else:
        scheduler = None

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