import torch
import torch.nn as nn
from datasets.ContinuousDataset import CardiacContinuous
from datasets.BatchDataset import CardiacBatch
from monai.metrics import DiceMetric
import monai.networks.utils as mutils

from active_dynamicmemory.ActiveDynamicMemoryModel import ActiveDynamicMemoryModel
import monai.networks.nets as monaimodels
import torchvision.models as models
import numpy as np

class CardiacActiveDynamicMemory(ActiveDynamicMemoryModel):

    def __init__(self, mparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        super(ActiveDynamicMemoryModel, self).__init__()
        self.collate_fn = None
        self.TaskDatasetBatch = CardiacBatch
        self.TaskDatasetContinuous = CardiacContinuous
        self.loss = nn.CrossEntropyLoss()
        self.init(mparams=mparams, modeldir=modeldir, device=device, training=training)

    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        """
        Load the cardiac segmentation model (Res. U-Net)
        :param droprate: dropout rate to be applied
        :param load_stylemodel: If true loads the style model (needed for training)
        :return: loaded model, stylemodel and gramlayers
        """
        model = monaimodels.UNet(dimensions=2, in_channels=1, out_channels=4,
                                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), norm='batch',
                                 dropout=droprate, num_res_units=2)

        if load_stylemodel:
            stylemodel = models.resnet50(pretrained=True)
            gramlayers = [stylemodel.layer2[-1].conv1]
            stylemodel.eval()

            return model, stylemodel, gramlayers

        return model, None, None

    def get_task_metric(self, image, target):
        """
        Task metric for cardiac segmentation is the dice score
        :param image: image the dice model should run on
        :param target: groundtruth segmentation of LV, MYO, RV
        :return: mean dice score metric
        """
        self.eval()
        x = image
        x = x[None, :].to(self.device)
        y = target[None, None, :, :].to(self.device)
        outy = self.model(x.float())
        y_hat_flat = torch.argmax(outy, dim=1)[:, None, :, :]

        one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
        one_hot_target = mutils.one_hot(y, num_classes=4)

        dm_m = DiceMetric(include_background=False, reduction='mean')

        dice, _ = dm_m(one_hot_pred, one_hot_target)
        self.train()
        return dice.detach().cpu().numpy()

    def completed_domain(self, m):
        """
        Domain is completed if m larger than a threshold
        :param m: value to compare to the threshold
        :return: Wheter or not the domain is considered completed
        """
        return m>self.mparams.completion_limit

    def validation_step(self, batch, batch_idx):
        """
        Validation step managed by pytorch lightning
        :param batch:
        :param batch_idx:
        :return: logs for validation
        """
        x, y, res, img = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        res = res[0]

        loss = self.loss(y_hat, y)
        y = y[:, None, :, :].to(self.device)
        y_hat_flat = torch.argmax(y_hat, dim=1)[:, None, :, :]

        one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
        one_hot_target = mutils.one_hot(y, num_classes=4)
        dm_b = DiceMetric(include_background=False, reduction='mean_batch')

        dice, _ = dm_b(one_hot_pred, one_hot_target)

        self.log_dict({f'val_loss_{res}': loss,
                       f'val_dice_lv_{res}': dice[0],
                       f'val_dice_myo_{res}': dice[1],
                       f'val_dice_rv_{res}': dice[2]})


    def get_task_loss(self, xs, ys):
        if type(xs) is list:
            loss = None
            for i, x in enumerate(xs):
                y = ys[i]

                y = torch.stack(y).to(self.device)
                x = x.to(self.device)
                y_hat = self.forward(x.float())
                if loss is None:
                    loss = self.loss(y_hat, y)
                else:
                    loss += self.loss(y_hat, y)
        else:
            if type(ys) is list:
                ys = torch.stack(ys).to(self.device)
            y_hat = self.forward(xs.float())
            loss = self.loss(y_hat, ys)

        return loss

    def get_uncertainties(self, x):
        y_hat_flats = []
        uncertainties = []

        for i in range(self.mparams.uncertainty_iterations):
            outy = self.forward(x)
            y_hat_flat = torch.argmax(outy, dim=1)[:, None, :, :]
            y_hat_flats.append(y_hat_flat)

        for i in range(len(x)):
            y_hat_detach = [t.detach().cpu().numpy() for t in y_hat_flats]
            uncertainties.append(np.quantile(np.array(y_hat_detach).std(axis=0)[i], 0.95))

        return uncertainties

