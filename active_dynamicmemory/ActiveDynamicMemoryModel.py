import argparse
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets.ContinuousDataset import BrainAgeContinuous, CardiacContinuous
from datasets.BatchDataset import BrainAgeBatch, CardiacBatch
from . import utils

from active_dynamicmemory.ActiveDynamicMemory import NaiveDynamicMemory, CasaDynamicMemory, UncertaintyDynamicMemory
from active_dynamicmemory.ActiveDynamicMemory import MemoryItem
from abc import ABC, abstractmethod

class ActiveDynamicMemoryModel(pl.LightningModule, ABC):

    def init(self, hparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        self.hparams = argparse.Namespace(**hparams)
        self.to(device)

        self.modeldir = modeldir

        self.model, self.stylemodel, self.gramlayers = self.load_model_stylemodel(self.hparams.droprate, load_stylemodel=training)
        if training:
            self.stylemodel.to(device)

        self.learning_rate = self.hparams.learning_rate
        self.train_counter = 0

        if self.hparams.continuous:
            self.budget = self.hparams.startbudget
            self.budgetchangecounter = 0
            if self.hparams.allowedlabelratio == 0:
                self.budgetrate = 0
            else:
                self.budgetrate = 1 / self.hparams.allowedlabelratio


        if not self.hparams.base_model is None and training:
            state_dict =  torch.load(os.path.join(modeldir, self.hparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith('model.'):
                    new_state_dict[key.replace('model.', '', 1)] = state_dict[key]
            self.model.load_state_dict(new_state_dict)

        # Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        self.scanner_checkpoints[self.hparams.order[0]] = True
        for scanner in self.hparams.order[1:]:
            self.scanner_checkpoints[scanner] = False

        if self.hparams.use_memory and self.hparams.continuous and training:
            self.init_memory_and_gramhooks()
        else:
            self.hparams.use_memory = False

    def init_memory_and_gramhooks(self):
        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        initelements = self.getmemoryitems_from_base(num_items=self.hparams.memorymaximum)
        print(initelements[0])

        if self.hparams.method == 'naive':
            self.trainingsmemory = NaiveDynamicMemory(initelements=initelements,
                                                      insert_rate=self.hparams.naive_insert_rate,
                                                      memorymaximum=self.hparams.memorymaximum,
                                                      seed=self.hparams.seed)
            self.insert_element = self.insert_element_naive

        elif self.hparams.method == 'casa':
            self.trainingsmemory = CasaDynamicMemory(initelements=initelements,
                                                      memorymaximum=self.hparams.memorymaximum,
                                                      seed=self.hparams.seed,
                                                      perf_queue_len=self.hparams.len_perf_queue,
                                                     transformgrams=self.hparams.transformgrams,
                                                     outlier_distance=self.hparams.outlier_distance)
            self.insert_element = self.insert_element_casa

        elif self.hparams.method == 'uncertainty':
            self.trainingsmemory = UncertaintyDynamicMemory(initelements=initelements,
                                                      memorymaximum=self.hparams.memorymaximum,
                                                      seed=self.hparams.seed,
                                                      perf_queue_len=self.hparams.len_perf_queue,
                                                      droprate=self.hparams.uncertainty_droprate)

            self.insert_element = self.insert_element_uncertainty

    def getmemoryitems_from_base(self, num_items=128):
        dl = DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                            iterations=None,
                                            batch_size=self.hparams.batch_size,
                                            split=['base']),
                            batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)

        memoryitems = []
        for batch in dl:
            torch.cuda.empty_cache()

            x, y, scanner, filepath = batch
            x = x.to(self.device)
            xstyle = torch.cat([x, x, x], dim=1)
            _ = self.stylemodel(xstyle)

            for i, f in enumerate(filepath):
                memoryitems.append(MemoryItem(x[i].detach().cpu(), y[i], f, scanner[i],
                                              current_grammatrix=self.grammatrices[0][
                                                  i].detach().cpu().numpy().flatten(),
                                              pseudo_domain=0))

            if len(memoryitems) >= num_items:
                break

            self.grammatrices = []

        return memoryitems[:num_items]

    def gram_hook(self,  m, input, output):
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
            new_mi = MemoryItem(img.detach().cpu(), y[i].detach().cpu(), filepath[i], scanner[i], grammatrix[0])
            self.trainingsmemory.insert_element(new_mi)

        return len(self.trainingsmemory.forceitems)!=0


    # This is called when hparams.method == 'style'
    def insert_element_casa(self, x, y, filepath, scanner):
        budget_before = self.budget

        for i, img in enumerate(x):
            grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
            new_mi = MemoryItem(img.detach().cpu(), y[i].detach().cpu(), filepath[i], scanner[i], grammatrix[0])
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
            print(self.trainingsmemory.domaincomplete)

            for k, v in self.trainingsmemory.domaincomplete.items():
                if not v:
                    if len(self.trainingsmemory.domainMetric[k]) == self.hparams.len_perf_queue:
                        mean_metric = np.mean(self.trainingsmemory.domainMetric[k])
                        if self.completed_domain(mean_metric):
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
                print('new shift to', shift_scanner)
                exp_name = utils.get_expname(self.hparams)
                weights_path = self.modeldir + exp_name + '_shift_' + shift_scanner + '.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.scanner_checkpoints[shift_scanner] = True

        if self.hparams.use_memory:
            #y = y[:, None]
            self.grammatrices = []
            xstyle = torch.cat([x, x, x], dim=1)
            _ = self.stylemodel(xstyle)

            train_a_step = self.insert_element(x, y, filepath, scanner)

            if train_a_step:
                x, y = self.trainingsmemory.get_training_batch(self.hparams.batch_size,
                                                                 batches=int(
                                                                     self.hparams.training_batch_size / self.hparams.batch_size))
                self.train_counter += 1
            else:
                return None

        loss = self.get_task_loss(x, y)
        self.log('train_log', loss)
        self.grammatrices = []
        return loss

    def forward(self, x):
        return self.model(x)

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




    @abstractmethod
    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        pass

    @abstractmethod
    def get_task_metric(self, image, target):
        pass

    @abstractmethod
    def completed_domain(self, m):
        pass

    @abstractmethod
    def get_task_loss(self, x, y):
        pass