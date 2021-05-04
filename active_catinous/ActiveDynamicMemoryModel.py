import argparse
import logging
import math
import os
import random
import skimage
import sklearn
from pprint import pprint
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pllogging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from models.AgePredictor import EncoderRegressor
from models.unet3d import EncoderModelGenesis

from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

from datasets.ContinuousDataset import BrainAgeContinuous, CardiacContinuous
from datasets.BatchDataset import BrainAgeBatch, CardiacBatch

from sklearn.metrics import mean_absolute_error

from scipy.spatial.distance import pdist, squareform

import monai.losses as mloss
from monai.metrics import compute_meandice, DiceMetric
import monai.networks.utils as mutils

from . import utils
import collections

from active_catinous.ActiveDynamicMemory import NaiveDynamicMemory, StyleDynamicMemory, UncertaintyDynamicMemory
from active_catinous.ActiveDynamicMemory import MemoryItem


class ActiveDynamicMemoryModel(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbose=False, training=True):
        super(ActiveDynamicMemoryModel, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)
        self.to(device)

        self.model, self.stylemodel, self.gramlayers = utils.load_model_stylemodel(self.hparams.task,
                                                                                   self.hparams.droprate,
                                                                                   stylemodel=training)

        self.learning_rate = self.hparams.learning_rate
        self.train_counter = 0

        self.budget = self.hparams.startbudget
        self.budgetchangecounter = 0
        if self.hparams.allowedlabelratio == 0:
            self.budgetrate = 0
        else:
            self.budgetrate = 1 / self.hparams.allowedlabelratio

        if self.hparams.task == 'brainage':
            self.loss = nn.MSELoss()
            self.mae = nn.L1Loss()
            self.TaskDataset = BrainAgeBatch
            self.get_task_error = self.get_absolute_error
        elif self.hparams.task == 'cardiac':
            #self.loss = mloss.DiceLoss(to_onehot_y=True, softmax=True) # TODO: or cross entropy as in nat comm?
            self.loss = nn.CrossEntropyLoss()
            self.TaskDataset = CardiacBatch #TODO: implement cardiac dataset
            self.get_task_error = self.get_dice_metric #TODO: implement!
            #self.mae = nn.L1Loss() #TODO: better loss?


        # Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        self.scanner_checkpoints[self.hparams.order[0]] = True
        for scanner in self.hparams.order[1:]:
            self.scanner_checkpoints[scanner] = False

        if self.hparams.use_memory and self.hparams.continuous and training:
            self.init_memory_and_gramhooks()
        else:
            self.hparams.use_memory = False

        if verbose:
            pprint(vars(self.hparams))


    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['datasetfile'] = '/project/catinous/brainds_split.csv'
        hparams['task'] = 'brainage'
        hparams['droprate'] = 0.0
        hparams['batch_size'] = 8
        hparams['training_batch_size'] = 8
        hparams['transition_phase_after'] = 0.8
        hparams['memorymaximum'] = 128
        hparams['use_memory'] = True
        hparams['balance_memory'] = True
        hparams['random_memory'] = False
        hparams['force_misclassified'] = True
        hparams['order'] = ['1.5T Philips', '3.0T Philips', '3.0T']
        hparams['continuous'] = True
        hparams['noncontinuous_steps'] = 3000
        hparams['noncontinuous_train_splits'] = ['base_train']
        hparams['val_check_interval'] = 100
        hparams['base_model'] = None
        hparams['run_postfix'] = '1'
        hparams['gram_weights'] = [1, 1, 1, 1]
        hparams['seed'] = 2314134
        hparams['completion_limit'] = 4.0
        hparams['gradient_clip_val'] = 0
        hparams['allowedlabelratio'] = 10
        hparams['scanner'] = None
        hparams['startbudget'] = 0.0
        hparams['len_perf_queue'] = 5

        return hparams

    def init_memory_and_gramhooks(self):
        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        initelements = self.getmemoryitems_from_base(num_items=self.hparams.memorymaximum)

        if self.hparams.method == 'naive':
            self.trainingsmemory = NaiveDynamicMemory(initelements=initelements,
                                                      insert_rate=self.hparams.naive_insert_rate,
                                                      memorymaximum=self.hparams.memorymaximum,
                                                      seed=self.hparams.seed)
            self.insert_element = self.insert_element_naive

        elif self.hparams.method == 'style':
            self.trainingsmemory = StyleDynamicMemory(initelements=initelements,
                                                      memorymaximum=self.hparams.memorymaxium,
                                                      seed=self.hparams.seed,
                                                      perf_queue_len=self.hparams.len_perf_queue)

            self.insert_element = self.insert_element_style

        elif self.hparams.method == 'uncertainty':
            self.trainingsmemory = UncertaintyDynamicMemory(initelements=initelements,
                                                      memorymaximum=self.hparams.memorymaxium,
                                                      seed=self.hparams.seed,
                                                      perf_queue_len=self.hparams.len_perf_queue,
                                                      droprate=self.hparams.uncertainty_droprate)

            self.insert_element = self.insert_element_uncertainty


    def get_absolute_error(self, image, label):
        self.eval()
        x = image
        x = x[None, :].to(self.device)
        outy = self.model(x.float())
        error = abs(outy.detach().cpu().numpy()[0]-label.numpy())
        self.train()
        return error

    def get_dice_metric(self, image, target):
        self.eval()
        x = image
        x = x[None, :].to(self.device)
        y = target[:, None, :, :].to(self.device)
        outy = self.model(x.float())
        y_hat_flat = torch.argmax(outy, dim=1)[:, None, :, :]

        one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
        one_hot_target = mutils.one_hot(y, num_classes=4)

        dice = compute_meandice(one_hot_pred, one_hot_target, include_background=False)
        self.train()
        return dice

    def getmemoryitems_from_base(self, num_items=128):
        dl = DataLoader(self.TaskDataset(self.hparams.datasetfile,
                                            iterations=None,
                                            batch_size=self.hparams.batch_size,
                                            split=['base_train']),
                            batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)

        memoryitems = []
        for batch in dl:
            torch.cuda.empty_cache()

            x, y, scanner, filepath = batch
            x = x.to(self.device)
            _ = self.stylemodel(x.float())

            for i, f in enumerate(filepath):
                memoryitems.append(MemoryItem(x[i].detach().cpu(), y[i], f, scanner[i],
                                              current_grammatrix=self.grammatrices[0][
                                                  i].detach().cpu().numpy().flatten(),
                                              pseudo_domain=0))

            if len(memoryitems) >= num_items:
                break

            self.grammatrices = []

        return memoryitems[:num_items]

    def gram_hook(self):
        if self.hparams.dim == 2:
            self.grammatrices.append(utils.gram_matrix(input[0]))
        elif self.hparams.dim == 3:
            self.grammatrices.append(utils.gram_matrix_3d(input[0]))
        else:
            raise NotImplementedError(f'gram hook with {self.hparams.dim} dimensions not defined')

    # This is called when hparams.method == 'naive'
    def insert_element_naive(self, x, y, filepath, scanner):
        for i, img in enumerate(x):
            grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
            new_mi = MemoryItem(img.detach().cpu(), y[i], filepath[i], scanner[i], grammatrix[0])
            self.trainingsmemory.insert_element(new_mi)

        return len(self.trainingsmemory.forceitems)!=0


    # This is called when hparams.method == 'style'
    def insert_element_style(self, x, y, filepath, scanner):
        budget_before = self.budget

        for i, img in enumerate(x):
            grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
            new_mi = MemoryItem(img.detach().cpu(), y[i], filepath[i], scanner[i], grammatrix[0])
            self.budget = self.trainingsmemory.insert_element(new_mi, self.budget, self)

        self.budget = self.trainingsmemory.check_outlier_memory(self.budget, self)
        self.trainingsmemory.counter_outlier_memory()

        if budget_before == self.budget:
            self.budgetchangecounter += 1
        else:
            self.budgetchangecounter = 1

        # form trainings X domain balanced batches to train one epoch on all newly inserted samples
        if not np.all(list(
                self.trainingsmemory.domaincomplete.values())) and self.budgetchangecounter < 10:  # only train when a domain is incomplete and new samples are inserted?
            for k, v in self.trainingsmemory.domaincomplete.items():
                if not v:
                    if len(self.trainingsmemory.domainError[k]) == self.hparams.len_perf_queue:
                        mean_error = np.mean(self.trainingsmemory.domainError[k])
                        print('domain', k, mean_error, self.trainingsmemory.domainError[k])
                        if mean_error < self.hparams.completion_limit:
                            self.trainingsmemory.domaincomplete[k] = True

            return True
        else:
            return False


    def training_step(self, batch, batch_idx):
        x, y, scanner, filepath = batch
        self.grammatrices = []

        if self.hparams.continuous:
            # save checkpoint at scanner shift
            newshift = False
            for s in scanner:
                if s != self.hparams.order[0] and not self.scanner_checkpoints[s]:
                    newshift = True
                    shift_scanner = s
            if newshift:
                exp_name = utils.get_expname(self.hparams)
                weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_' + shift_scanner + '.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.scanner_checkpoints[shift_scanner] = True

        if self.hparams.use_memory:
            y = y[:, None]
            self.grammatrices = []
            _ = self.stylemodel(x.float())

            train_a_step = self.insert_element(x, y, filepath, scanner)

            if train_a_step:
                xs, ys = self.trainingsmemory.get_training_batch(self.hparams.batch_size,
                                                                 batches=int(
                                                                     self.hparams.training_batch_size / self.hparams.batch_size))

                loss = None
                for i, x in enumerate(xs):
                    y = ys[i]

                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x.float())
                    if loss is None:
                        loss = self.loss(y_hat, y)
                    else:
                        loss += self.loss(y_hat, y)

                self.train_counter += 1
                self.log('train_loss', loss)
                return loss
            else:
                return None

        else:
            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y)
            self.log('train_loss', loss)
            return loss

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y, res, img = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        res = res[0]
        if self.hparams.task=='brainage':
            self.log_dict({f'val_loss_{res}': self.loss(y_hat, y),
                       f'val_mae_{res}': self.mae(y_hat, y)}) #TODO: MAE can only work for brain age not cardiac
        elif self.hparams.task=='cardiac':
            loss = self.loss(y_hat, y)
            y = y[:, None, :, :].to(self.device)
            y_hat_flat = torch.argmax(y_hat, dim=1)[:, None, :, :]

            one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
            one_hot_target = mutils.one_hot(y, num_classes=4)
            dm_b = DiceMetric(include_background=False, reduction='mean_batch')

            dice, _ = dm_b(one_hot_pred, one_hot_target)

            self.log_dict({f'val_loss_{res}': loss,
                           f'val_dice_1_{res}': dice[0],
                           f'val_dice_2_{res}': dice[1],
                           f'val_dice_3_{res}': dice[2]})


    #def test_step(self, batch, batch_idx):
    #    x, y, res, img = batch
    #    self.grammatrices = []

    #    y_hat = self.forward(x.float())

    #    res = res[0]
    #    return {f'val_loss_{res}': self.loss(y_hat, y[:, None].float()),
    #            f'val_mae_{res}': self.mae(y_hat, y[:, None].float())} #TODO: MAE can only work for brain age not cardiac

    #def test_end(self, outputs):
    #    val_mean = dict() #TODO: MAE can only work for brain age not cardiac
    #    res_count = dict()

    #    for output in outputs:

    #        for k in output.keys():
    #            if k not in val_mean.keys():
    #                val_mean[k] = 0
    #                res_count[k] = 0

    #            val_mean[k] += output[k]
    #            res_count[k] += 1

    #    tensorboard_logs = dict()
    #    for k in val_mean.keys():
    #        tensorboard_logs[k] = val_mean[k] / res_count[k]

    #    return {'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        if self.hparams.continuous:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeContinuous(self.hparams.datasetfile,
                                                                   transition_phase_after=self.hparams.transition_phase_after,
                                                     seed=self.hparams.seed,
                                                 order=self.hparams.order),
                                  batch_size=self.hparams.batch_size, num_workers=8, drop_last=True)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacContinuous(self.hparams.datasetfile,
                                                 transition_phase_after=self.hparams.transition_phase_after,
                                                    seed=self.hparams.seed,
                                                 order=self.hparams.order),
                                  batch_size=self.hparams.batch_size, num_workers=8, drop_last=True)
        else:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=8)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits,
                                                  res=self.hparams.scanner),
                                  batch_size=self.hparams.batch_size, num_workers=8)

    #@pl.data_loader
    def val_dataloader(self):
        if self.hparams.task == 'brainage':
            return DataLoader(BrainAgeBatch(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)
        elif self.hparams.task == 'cardiac':
            return DataLoader(CardiacBatch(self.hparams.datasetfile,
                                          split=['val']),
                          batch_size=4,
                          num_workers=1)



def trained_model(hparams, train=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = ActiveDynamicMemoryModel(hparams=hparams, device=device, training=train)
    exp_name = utils.get_expname(model.hparams)
    weights_path = utils.TRAINED_MODELS_FOLDER + exp_name +'.pt'
    print(weights_path)
    if not os.path.exists(weights_path) and train:
        logger = pllogging.TestTubeLogger(utils.LOGGING_FOLDER, name=exp_name)
        trainer = Trainer(gpus=1, max_epochs=1, logger=logger,
                          val_check_interval=model.hparams.val_check_interval,
                          gradient_clip_val=model.hparams.gradient_clip_val,
                          checkpoint_callback=False)
        trainer.fit(model)
        model.freeze()
        torch.save(model.state_dict(), weights_path)
        if model.hparams.continuous:
            print('train counter', model.train_counter)
            print('label counter', model.trainingsmemory.labeling_counter)
        if model.hparams.continuous and model.hparams.use_memory:
            utils.save_memory_to_csv(model.trainingsmemory.memorylist, utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv')
    elif os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'):
        print('Read: ' + weights_path)
        state_dict = torch.load(weights_path)
        new_state_dict = dict()
        for k in state_dict.keys():
            if k.startswith('model.'):
                new_state_dict[k.replace("model.", "", 1)] = state_dict[k]
        model.model.load_state_dict(new_state_dict)
        model.freeze()
    else:
        print(weights_path, 'does not exist')
        model = None
        return model, None, None, exp_name + '.pt'

    if model.hparams.continuous and model.hparams.use_memory:
        if os.path.exists(utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv'):
            df_memory = pd.read_csv(utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv')
        else:
            df_memory = None
    else:
        df_memory=None

    # always get the last version
    try:
        max_version = max([int(x.split('_')[1]) for x in os.listdir(utils.LOGGING_FOLDER + exp_name)])
        logs = pd.read_csv(utils.LOGGING_FOLDER + exp_name + '/version_{}/metrics.csv'.format(max_version))
    except Exception as e:
        print(e)
        logs = None

    return model, logs, df_memory, exp_name +'.pt'

def is_cached(hparams):
    #model = FastGramDynamicMemoryBrainAge(hparams=hparams)
    hparams = utils.default_params(ActiveDynamicMemoryModel.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    #model = FastGramDynamicMemoryBrainAge(hparams=hparams)
    hparams = utils.default_params(ActiveDynamicMemoryModel.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'