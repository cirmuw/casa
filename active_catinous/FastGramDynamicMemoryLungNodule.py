import argparse
import logging
import math
import os
import random
import skimage
import sklearn
from pprint import pprint
import numpy as np
import models.MDTRetinaNet as mdtr

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pllogging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import models.RetinaNetDetection as retinanet

from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

from datasets.MDTLUNADatasetContinuous import MDTLUNADatasetContinuous
from datasets.MDTLUNADataset import MDTLUNADataset

from sklearn.metrics import mean_absolute_error

from scipy.spatial.distance import pdist, squareform
import torchvision.models as tvmodels

from . import utils
import collections

class FastGramDynamicMemoryLungNodule(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbose=True, training=True):
        super(FastGramDynamicMemoryLungNodule, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)

        #print all parameters
        if verbose:
            for k in self.hparams:
                print(k, self.hparams[k])

        #read settings from hparams
        print('reading settings from hparams')
        if 'naive_continuous' in self.hparams:
            self.naive_continuous = True
        else:
            self.naive_continuous = False
        self.learning_rate = self.hparams.learning_rate
        self.budget = self.hparams.startbudget

        #init counters
        print('init')
        self.train_counter = 0
        self.budgetchangecounter = 0
        if self.hparams.allowedlabelratio==0:
            self.budgetrate = 0
        else:
            self.budgetrate = 1/self.hparams.allowedlabelratio
        self.shiftcheckpoint_1 = False
        self.shiftcheckpoint_2 = False
        self.loss = nn.MSELoss()
        self.mae = nn.L1Loss()

        #init task model
        print('init task model')
        modelcf = mdtr.config(n_slices=self.hparams.n_slices)

        modellogger = logging.getLogger('medicaldetectiontoolkit')
        modellogger.setLevel(logging.DEBUG)
        self.model = mdtr.net(modelcf, modellogger)

        if not self.hparams.base_model is None:
            print('read base model')
            state_dict =  torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace('model.', '')] = state_dict[key]
            self.model.load_state_dict(new_state_dict)

        self.model.to(device)
        self.to(device)

        if not training:
            print('only validation, no training model')
        else:
            print('init for training')
            #init style model
            self.stylemodel = tvmodels.resnet50(pretrained=True)
            self.stylemodel.to(device)
            self.stylemodel.eval()
            if self.hparams.use_memory and self.hparams.continuous:
                self.init_cache_and_gramhooks()
                initmemoryelements = self.getmemoryitems_from_base(num_items=self.hparams.memorymaximum)

                if self.naive_continuous:
                    print('init naive memory')
                    self.trainingsmemory = NaiveDynamicMemoryLN(initelements=initmemoryelements,
                                                                insert_rate=self.hparams.naive_continuous_rate,
                                                                memorymaximum=self.hparams.memorymaximum,
                                                                gram_weights=self.hparams.gram_weights,
                                                                seed=self.hparams.seed)
                else:
                    print('init CASA memory')
                    self.trainingsmemory = DynamicMemoryLN(initelements=initmemoryelements,
                                                           memorymaximum=self.hparams.memorymaximum,
                                                           gram_weights=self.hparams.gram_weights,
                                                           seed=self.hparams.seed,
                                                           perf_queue_len=self.hparams.len_perf_queue)

                    print(len(self.trainingsmemory.get_domainitems(0)))
            else:
                self.hparams.use_memory = False

    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['datasetfile'] = '/project/catinous/lunadata/luna_lunacombined_dataset.csv'
        hparams['batch_size'] = 8
        hparams['training_batch_size'] = 8
        hparams['transition_phase_after'] = 0.8
        hparams['memorymaximum'] = 128
        hparams['use_memory'] = True
        hparams['balance_memory'] = True
        hparams['random_memory'] = False
        hparams['force_misclassified'] = True
        hparams['order'] = ['ges', 'geb', 'sie']
        hparams['continuous'] = True
        hparams['noncontinuous_steps'] = 3000
        hparams['noncontinuous_train_splits'] = ['base_train']
        hparams['val_check_interval'] = 100
        hparams['base_model'] = None
        hparams['run_postfix'] = '1'
        hparams['gram_weights'] = [1, 1, 1, 1]
        hparams['seed'] = 2314134
        hparams['completion_limit'] = 4.0 #TODO: set this after baselines
        hparams['gradient_clip_val'] = 0
        hparams['allowedlabelratio'] = 10
        hparams['scanner'] = None
        hparams['startbudget'] = 0.0
        hparams['len_perf_queue'] = 5
        hparams['n_slices'] = 1

        return hparams

    def getmemoryitems_from_base(self, num_items=128):
        dl = DataLoader(MDTLUNADataset(self.hparams.datasetfile,
                                   iterations=None,
                                   batch_size=self.hparams.batch_size,
                                   split=['base_train']),
                   batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True, shuffle=True)

        memoryitems = []
        for batch in dl:
            torch.cuda.empty_cache()

            x, y, filepath, scanner = batch
            x = x.to(self.device)
            y_style = self.stylemodel(x.float())

            for i, f in enumerate(filepath):
                memoryitems.append(MemoryItem(x[i].detach().cpu(), y[i], f, scanner[i],
                                              current_grammatrix=self.grammatrices[0][i].detach().cpu().numpy().flatten(),
                                              pseudo_domain=0))

            if len(memoryitems)>=num_items:
                break

            self.grammatrices = []
        return memoryitems[:num_items]


    def init_cache_and_gramhooks(self):
        self.grammatrices = []
        self.gramlayers = [
            self.stylemodel.layer2[-1].conv1]

        #register hooks
        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)
        logging.info('Gram hooks and memory initialized. Cachesize: %i' % self.hparams.memorymaximum)

    def gram_hook(self, m, input, output):
        self.grammatrices.append(utils.gram_matrix(input[0]))

    def training_step(self, batch, batch_idx):
        x, y, filepath, scanner = batch

        self.budget += len(filepath) * self.budgetrate

        if self.hparms.order[1] in scanner and not self.shiftcheckpoint_1:
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_1_ckpt.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.shiftcheckpoint_1 = True
        if self.hparms.order[2] in scanner and not self.shiftcheckpoint_2:
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_2_ckpt.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.shiftcheckpoint_2 = True

        if not self.naive_continuous and self.hparams.use_memory:
                torch.cuda.empty_cache()
                y = y[:, None]
                self.grammatrices = []
                y_style = self.stylemodel(x.float())

                budget_before = self.budget

                for i, img in enumerate(x):
                    grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
                    new_mi = MemoryItem(img.detach().cpu(), y[i], filepath[i], scanner[i], grammatrix[0])
                    self.budget = self.trainingsmemory.insert_element(new_mi, self.budget, self)

                self.budget = self.trainingsmemory.check_outlier_memory(self.budget, self)
                self.trainingsmemory.counter_outlier_memory()

                if budget_before==self.budget:
                    self.budgetchangecounter+=1
                else:
                    self.budgetchangecounter=1

                if not np.all(list(self.trainingsmemory.domaincomplete.values())) and self.budgetchangecounter<10:
                    for k, v in self.trainingsmemory.domaincomplete.items():
                        if not v:
                            if len(self.trainingsmemory.domainMAE[k])==self.hparams.len_perf_queue:
                                meanperf = np.mean(self.trainingsmemory.domainPerf[k])
                                print('domain', k, meanperf, self.trainingsmemory.domainPerf[k])
                                if meanperf<self.hparams.completion_limit: #TODO < or > depending on metric
                                    self.trainingsmemory.domaincomplete[k] = True

                    xs, ys = self.trainingsmemory.get_training_batch(self.hparams.batch_size,
                                                                     batches=int(self.hparams.training_batch_size/self.hparams.batch_size))

                    loss = None
                    for i, x in enumerate(xs):
                        y = ys[i]

                        x = x.to(self.device)
                        y = y.to(self.device)

                        y_hat = self.model(x.float())
                        if loss is None:
                            loss = self.loss(y_hat, y.float())
                        else:
                            loss += self.loss(y_hat, y.float())

                    self.train_counter += 1
                    self.log('train_loss', loss)

                    return loss
                else:
                    return None
        elif self.naive_continuous and self.hparams.use_memory:
            y = y[:, None]
            self.grammatrices = []
            y_style = self.stylemodel(x.float())

            for i, img in enumerate(x):
                grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
                new_mi = MemoryItem(img.detach().cpu(), y[i], filepath[i], scanner[i], grammatrix[0])
                self.trainingsmemory.insert_element(new_mi)

            if len(self.trainingsmemory.forceitems)!=0:
                xs, ys = self.trainingsmemory.get_training_batch(self.hparams.batch_size,
                                                                 batches=int(self.hparams.training_batch_size/self.hparams.batch_size))

                loss = None
                for i, x in enumerate(xs):
                    y = ys[i]

                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x.float())
                    if loss is None:
                        loss = self.loss(y_hat, y.float())
                    else:
                        loss += self.loss(y_hat, y.float())

                self.train_counter += 1
                self.log('train_loss', loss)
                return loss
            else:
                return None
        else:
            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y[:, None].float())
            self.log('train_loss', loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y, img, res = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        res = res[0]
        self.log_dict({f'val_loss_{res}': self.loss(y_hat, y[:, None].float()),
                       f'val_mae_{res}': self.mae(y_hat, y[:, None].float())})

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, y, img, res = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        res = res[0]
        return {f'val_loss_{res}': self.loss(y_hat, y[:, None].float()),
                f'val_mae_{res}': self.mae(y_hat, y[:, None].float())}

    def test_end(self, outputs):
        val_mean = dict()
        res_count = dict()

        for output in outputs:

            for k in output.keys():
                if k not in val_mean.keys():
                    val_mean[k] = 0
                    res_count[k] = 0

                val_mean[k] += output[k]
                res_count[k] += 1

        tensorboard_logs = dict()
        for k in val_mean.keys():
            tensorboard_logs[k] = val_mean[k] / res_count[k]

        return {'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def get_absolute_error(self, image, label):
        self.eval()
        x = image
        x = x[None, :].to(self.device)
        outy = self.model(x.float())
        error = abs(outy.detach().cpu().numpy()[0]-label.numpy())
        self.train()
        return error


    #@pl.data_loader
    def train_dataloader(self):
        if self.hparams.continuous:
            return DataLoader(MDTLUNADatasetContinuous(self.hparams.datasetfile,
                                                               transition_phase_after=self.hparams.transition_phase_after,
                                                                seed=self.hparams.seed),
                              batch_size=self.hparams.batch_size, num_workers=4, drop_last=True, pin_memory=False)
        else:
            return DataLoader(MDTLUNADataset(self.hparams.datasetfile,
                                              iterations=self.hparams.noncontinuous_steps,
                                              batch_size=self.hparams.batch_size,
                                              split=self.hparams.noncontinuous_train_splits,
                                              res=self.hparams.scanner,
                                              seed=self.hparams.seed
                                              ),
                              batch_size=self.hparams.batch_size, num_workers=4, pin_memory=False)

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(MDTLUNADataset(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=2, pin_memory=False, drop_last=False)

class DynamicMemoryLN():

    def __init__(self, initelements, memorymaximum=256, seed=None, transformgrams=True, perf_queue_len=5):
        self.memoryfull = False
        self.memorylist = initelements
        self.memorymaximum = memorymaximum

        self.samples_per_domain = memorymaximum
        self.domaincounter = {0: len(self.memorylist)} #0 is the base training domain
        self.max_per_domain = memorymaximum
        self.seed = seed
        self.labeling_counter = 0

        graminits = []
        for mi in initelements:
            graminits.append(mi.current_grammatrix)

        if transformgrams:
            self.transformer = SparseRandomProjection(random_state=seed, n_components=30)
            self.transformer.fit(graminits)
            for mi in initelements:
                mi.current_grammatrix = self.transformer.transform(mi.current_grammatrix.reshape(1, -1))
            trans_initelements = self.transformer.transform(graminits)
        else:
            self.transformer = None
            trans_initelements = graminits

        clf = IsolationForest(n_estimators=10, random_state=seed).fit(trans_initelements)
        self.isoforests = {0: clf}

        self.domaincomplete = {0: True}

        self.domainPerf = {0: collections.deque(maxlen=perf_queue_len)} #TODO: this is an arbritary threshold
        self.perf_queue_len = perf_queue_len
        self.outlier_memory = []
        self.outlier_epochs = 25 #TODO: this is an arbritary threshold

        self.img_size = (64, 128, 128)

    def check_outlier_memory(self, budget, model):
        if len(self.outlier_memory)>5 and int(budget)>=5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]

            distances = squareform(pdist(outlier_grams))
            if sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5]<0.20: #TODO: this is an arbritary threshold

                clf = IsolationForest(n_estimators=5, random_state=self.seed, warm_start=True, contamination=0.10).fit(
                    outlier_grams)

                new_domain_label = len(self.isoforests)
                self.domaincomplete[new_domain_label] = False
                self.domaincounter[new_domain_label] = 0
                self.domainPerf[new_domain_label] = collections.deque(maxlen=self.perf_queue_len)
                self.max_per_domain = int(self.memorymaximum/(new_domain_label+1))

                self.flag_items_for_deletion()

                to_delete = []
                for k, p in enumerate(clf.predict(outlier_grams)):
                    if int(budget)>0:
                        if p == 1:
                            idx = self.find_insert_position()
                            if idx != -1:
                                elem = self.outlier_memory[k]
                                elem.pseudo_domain = new_domain_label
                                self.memorylist[idx] = elem
                                self.domaincounter[new_domain_label] += 1
                                to_delete.append(self.outlier_memory[k])
                                budget -= 1.0
                                self.labeling_counter += 1
                                self.domainPerf[new_domain_label].append(model.get_absolute_error(elem.img, elem.label))
                    else:
                        print('run out of budget ', budget)
                for elem in to_delete:
                    self.outlier_memory.remove(elem)

                self.isoforests[new_domain_label] = clf

                for elem in self.get_domainitems(new_domain_label):
                    print('found new domain', new_domain_label, elem.scanner)

        return budget

    def find_insert_position(self):
        for idx, item in enumerate(self.memorylist):
            if item.deleteflag:
                return idx
        return -1

    def flag_items_for_deletion(self):
        for k, v in self.domaincomplete.items():
            domain_count = len(self.get_domainitems(k))
            if domain_count>self.max_per_domain:
                todelete = domain_count-self.max_per_domain
                for item in self.memorylist:
                    if todelete>0:
                        if item.pseudo_domain==k:
                            if not item.deleteflag:
                                item.deleteflag = True

                            todelete -= 1


    def counter_outlier_memory(self):
        for item in self.outlier_memory:
            item.counter += 1
            if item.counter>self.outlier_epochs:
                self.outlier_memory.remove(item)

    def insert_element(self, item, budget, model):
        if self.transformer is not None:
            item.current_grammatrix = self.transformer.transform(item.current_grammatrix.reshape(1, -1))
            item.current_grammatrix = item.current_grammatrix[0]

        domain = self.check_pseudodomain(item.current_grammatrix)
        item.pseudo_domain = domain

        if domain==-1:
            #insert into outlier memory
            #check outlier memory for new clusters
            self.outlier_memory.append(item)
        else:
            if not self.domaincomplete[domain] and int(budget)>0:
                #insert into dynamic memory and training
                idx = self.find_insert_position()
                if idx == -1: # no free memory position, replace an element already in memory
                    mingramloss = 1000
                    for j, mi in enumerate(self.memorylist):
                        if mi.pseudo_domain == domain:
                            loss = F.mse_loss(torch.tensor(item.current_grammatrix), torch.tensor(mi.current_grammatrix), reduction='mean')

                            if loss < mingramloss:
                                mingramloss = loss
                                idx = j
                    print(self.memorylist[idx].scanner, 'replaced by', item.scanner, 'in domain', domain)
                else:
                    self.domaincounter[domain] += 1
                self.memorylist[idx] = item
                self.labeling_counter += 1
                self.domainPerf[domain].append(model.get_absolute_error(item.img, item.label))

                # add tree to clf of domain
                clf = self.isoforests[domain]
                domain_items = self.get_domainitems(domain)
                domain_grams = [d.current_grammatrix for d in domain_items]

                if len(clf.estimators_) < 10:
                    n_estimators = len(clf.estimators_) + 1
                    clf.__setattr__('n_estimators', n_estimators)
                else:
                    clf = IsolationForest(n_estimators=10)

                clf.fit(domain_grams)
                self.isoforests[domain] = clf

                budget -= 1.0
            else:
                if int(budget)<1:
                    print('run out of budget ', budget)

        return budget

    def check_pseudodomain(self, grammatrix):
        max_pred = 0
        current_domain = -1

        for j, clf in self.isoforests.items():
            current_pred = clf.decision_function(grammatrix.reshape(1, -1))

            if current_pred>max_pred:
                current_domain = j
                max_pred = current_pred

        return current_domain

    def get_training_batch(self, batchsize, batches=1):
        xs = []
        ys = []

        to_force = []

        half_batch = int(batchsize/2)

        for d, c in self.domaincomplete.items():
            if not c:
                for mi in self.memorylist:
                    if mi.pseudo_domain == d:
                        to_force.append(mi)

        for b in range(batches):
            j = 0
            bs = batchsize
            x = torch.empty(size=(batchsize, 1, self.img_size[0], self.img_size[1], self.img_size[2]))
            y = torch.empty(size=(batchsize, 1))

            random.shuffle(to_force)
            for mi in to_force[-half_batch:]:
                if j<bs:
                    x[j] = mi.img
                    y[j] = mi.label
                    j += 1
                    mi.traincounter += 1

            bs -= j
            if bs>0:
                random.shuffle(self.memorylist)
                for mi in self.memorylist[-bs:]:
                    x[j] = mi.img
                    y[j] = mi.label
                    j += 1
                    mi.traincounter += 1

            xs.append(x)
            ys.append(y)

        return (xs, ys)

    def get_domainitems(self, domain):
        items = []
        for mi in self.memorylist:
            if mi.pseudo_domain == domain:
                items.append(mi)
        return items

class NaiveDynamicMemoryLN():

    def __init__(self, initelements, memorymaximum=256, insert_rate=10, gram_weights=None, seed=None):
        self.memoryfull = False
        self.memorylist = initelements
        self.memorymaximum = memorymaximum
        self.gram_weigths = gram_weights
        self.seed = seed
        self.insert_counter = 0
        self.insert_rate = insert_rate
        self.forceitems = []
        self.labeling_counter = 0
        self.img_size = (64, 128, 128)


    def insert_element(self, item):
        self.insert_counter += 1

        if self.insert_counter%self.insert_rate==0:
            if len(self.memorylist)<self.memorymaximum:
                self.memorylist.append(item)
                self.forceitems.append(item)
                self.labeling_counter += 1
            else:
                assert (item.current_grammatrix is not None)
                insertidx = -1
                mingramloss = 1000
                for j, mi in enumerate(self.memorylist):
                    loss = F.mse_loss(torch.tensor(item.current_grammatrix), torch.tensor(mi.current_grammatrix),
                                      reduction='mean')

                    if loss < mingramloss:
                        mingramloss = loss
                        insertidx = j
                self.memorylist[insertidx] = item
                self.forceitems.append(item)
                self.labeling_counter += 1

    def get_training_batch(self, batchsize, batches=1):
        xs = []
        ys = []

        half_batch = int(batchsize / 2)

        for b in range(batches):
            j = 0
            bs = batchsize
            x = torch.empty(size=(batchsize, 1, self.img_size[0], self.img_size[1], self.img_size[2]))
            y = torch.empty(size=(batchsize, 1))

            if len(self.forceitems)>0:
                random.shuffle(self.forceitems)
                m = min(len(self.forceitems), half_batch)
                for mi in self.forceitems[-m:]:
                    if j<bs:
                        x[j] = mi.img
                        y[j] = mi.label
                        j += 1
                        mi.traincounter += 1

            bs -= j
            if bs>0:
                random.shuffle(self.memorylist)
                for mi in self.memorylist[-bs:]:
                    x[j] = mi.img
                    y[j] = mi.label
                    j += 1
                    mi.traincounter += 1

            xs.append(x)
            ys.append(y)

        self.forceitems = []

        return (xs, ys)


class MemoryItem():

    def __init__(self, img, label, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.label = label.detach().cpu()
        self.filepath = filepath
        self.scanner = scanner
        self.counter = 0
        self.traincounter = 0
        self.deleteflag = False
        self.pseudo_domain = pseudo_domain
        self.current_grammatrix = current_grammatrix


def trained_model(hparams, train=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = FastGramDynamicMemoryBrainAge(hparams=hparams, device=device, for_training=train)
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
                new_state_dict[k.replace("model.", "")] = state_dict[k]
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
    hparams = utils.default_params(FastGramDynamicMemoryBrainAge.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    #model = FastGramDynamicMemoryBrainAge(hparams=hparams)
    hparams = utils.default_params(FastGramDynamicMemoryBrainAge.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'