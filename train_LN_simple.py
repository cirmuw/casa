import models.MDTRetinaNet as mdtr
import logging
import torch
from datasets.MDTLUNADataset import MDTLUNADataset
from datasets.LUNADataset import LUNADataset

from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import models.MDTRetinaNet as mdtr
import models.RetinaNetDetection as rnd



def train_loop(ds_path, ds_split, savepath, lr=1e-4, scheduler_steps=10, n_slices=1, epochs=50, batch_size=4,
               labelDebug=None, operate_stride1=False):
    device = torch.device('cuda')
    cf = mdtr.config(n_slices=n_slices, operate_stride1=operate_stride1)
    logger = logging.getLogger('medicaldetectiontoolkit')
    logger.setLevel(logging.DEBUG)
    model = mdtr.net(cf, logger)

    model.to(device)

    ds = MDTLUNADataset(ds_path, split=ds_split, n_slices=n_slices, labelDebug=labelDebug)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)
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
    if savepath is not None:
        torch.save(model.state_dict(), savepath)

def train_loop_simple(ds_path, ds_split, savepath, lr=1e-4, scheduler_steps=10, n_slices=1, epochs=50, batch_size=4,
               labelDebug=None, operate_stride1=False):
    device = torch.device('cuda')

    print('simple model!')
    model = rnd.resnet50(num_classes=2, pretrained=False).to(device)

    model.to(device)

    ds = LUNADataset(ds_path, split=ds_split, labelDebug=labelDebug)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler_steps is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_steps, gamma=0.1)
    else:
        scheduler = None

    for epoch in range(epochs):
        running_loss = 0.0
        running_class = 0.0
        running_regress = 0.0
        for iter_num, data in enumerate(dl):
            optimizer.zero_grad()

            img = torch.FloatTensor(data[0].float()).to(device)
            classification_loss, regression_loss = model([img, data[1].float().to(device)])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            running_loss += loss.item()
            running_class += classification_loss.item()
            running_regress += regression_loss.item()

        print(epoch, 'loss:', running_loss / len(dl), 'class:', running_regress / len(dl), 'box:',
              running_regress / len(dl))
        if scheduler is not None:
            scheduler.step()
    if savepath is not None:
        torch.save(model.state_dict(), savepath)