import torch
import torch.nn as nn
from datasets.ContinuousDataset import BrainAgeContinuous
from datasets.BatchDataset import BrainAgeBatch
from active_dynamicmemory.ActiveDynamicMemoryModel import ActiveDynamicMemoryModel
import models.AgePredictor as agemodels
from models.unet3d import EncoderModelGenesis
import numpy as np

class BrainAgeActiveDynamicMemory(ActiveDynamicMemoryModel):

    def __init__(self, hparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        super(ActiveDynamicMemoryModel, self).__init__()
        self.collate_fn = None
        self.TaskDatasetBatch = BrainAgeBatch
        self.TaskDatasetContinuous = BrainAgeContinuous

        self.mae = nn.L1Loss()
        self.loss = nn.MSELoss()
        self.init(hparams=hparams, modeldir=modeldir, device=device, training=training)


    def get_task_metric(self, image, target):
        """
        Task metric for brain age estimation is the absolute error
        :param image: image the absolute error should be calculted for
        :param target: real age of the patient
        :return: absolute error of estimation to real age
        """
        self.eval()
        x = image
        x = x[None, :].to(self.device)
        outy = self.model(x.float())
        error = abs(outy.detach().cpu().numpy()[0] - target.numpy())
        self.train()
        return error


    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        """
        Load the cardiac segmentation model (Res. U-Net)
        :param droprate: dropout rate to be applied
        :param load_stylemodel: If true loads the style model (needed for training)
        :return: loaded model, stylemodel and gramlayers
        """
        model = agemodels.EncoderRegressor(droprate=droprate)

        if load_stylemodel:
            stylemodel = EncoderModelGenesis()
            # Load pretrained model genesis
            weight_dir = 'models/Genesis_Chest_CT.pt'
            checkpoint = torch.load(weight_dir)
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                if key.startswith('module.down_'):
                    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            stylemodel.load_state_dict(unParalled_state_dict)
            gramlayers = [stylemodel.down_tr64.ops[1].conv1]
            stylemodel.eval()

            return model, stylemodel, gramlayers

        return model, None, None

    def completed_domain(self, m):
        """
        Domain is completed if m smaller than a threshold
        :param m: value to compare to the threshold
        :return: Wheter or not the domain is considered completed
        """
        return m<self.hparams.completion_limit

    def validation_step(self, batch, batch_idx):
        """
        Validation step managed by pytorch lightning
        :param batch:
        :param batch_idx:
        :return: logs for validation
        """
        x, y, res, img = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())[:, 0]

        res = res[0]
        self.log_dict({f'val_loss_{res}': self.loss(y_hat, y),
                   f'val_mae_{res}': self.mae(y_hat, y)})

    def get_task_loss(self, xs, ys):
        if type(xs) is list:
            loss = None
            for i, x in enumerate(xs):
                y = ys[i]

                x = x.to(self.device)
                y = torch.tensor(y).to(self.device)
                y_hat = self.forward(x.float())[:, 0]
                if loss is None:
                    loss = self.loss(y_hat, y)
                else:
                    loss += self.loss(y_hat, y)
        else:
            y_hat = self.forward(xs.float())[:, 0]
            loss = self.loss(y_hat, ys)

        return loss

    def get_uncertainties(self, x):
        output = []

        for i in range(self.hparams.uncertainty_iterations):
            outy = self.forward(x)
            output.append([o[0] for o in outy.detach().cpu().numpy()])

        return np.array(output).std(axis=0)