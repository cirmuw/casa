import argparse
import logging
import math
import os
import random
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

from datasets.BrainAgeContinuous import BrainAgeContinuous
from datasets.BrainAgeDataset import BrainAgeDataset

from sklearn.metrics import mean_absolute_error

from scipy.spatial.distance import pdist, squareform

from . import utils

class FastGramDynamicMemoryBrainAge(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbose=False):
        super(FastGramDynamicMemoryBrainAge, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)

        self.learning_rate = self.hparams.learning_rate
        self.train_counter = 0

        self.budget = 10.0
        self.budgetrate = 1/self.hparams.allowedlabelratio

        self.to(device)
        print('init')

        self.stylemodel = EncoderModelGenesis()

        # Load pretrained model genesis
        weight_dir = 'models/Genesis_Chest_CT.pt'
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            if key.startswith('module.down_'):
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        self.stylemodel.load_state_dict(unParalled_state_dict)

        if self.hparams.use_memory and self.hparams.continuous:
            self.init_cache_and_gramhooks()
        else:
            if verbose:
                logging.info('No continous learning, following parameters are invalidated: \n'
                             'transition_phase_after \n'
                             'cachemaximum \n'
                             'use_cache \n'
                             'random_cache \n'
                             'force_misclassified \n'
                             'direction')
            self.hparams.use_memory = False

        self.stylemodel.to(device)
        self.stylemodel.eval()

        self.model = EncoderRegressor()
        self.model.to(device)
        if not self.hparams.base_model is None:
            state_dict =  torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace('model.', '')] = state_dict[key]
            self.model.load_state_dict(new_state_dict)


        self.loss = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.shiftcheckpoint_1 = False
        self.shiftcheckpoint_2 = False

        if self.hparams.continuous:

            initmemoryelements = self.getmemoryitems_from_base(num_items=self.hparams.memorymaximum)

            #PREFILL memory with base training samples!!!!
            self.trainingsmemory = DynamicMemoryAge(initelements=initmemoryelements,
                                                    memorymaximum=self.hparams.memorymaximum,
                                                   gram_weights=self.hparams.gram_weights,
                                                    seed=self.hparams.seed)

            print(len(self.trainingsmemory.get_domainitems(0)))

        if verbose:
            pprint(vars(self.hparams))

    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['datasetfile'] = '/project/catinous/brainds_split.csv'
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

        return hparams

    def getmemoryitems_from_base(self, num_items=128):
        dl = DataLoader(BrainAgeDataset(self.hparams.datasetfile,
                                   iterations=None,
                                   batch_size=self.hparams.batch_size,
                                   split=['base_train']),
                   batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)

        memoryitems = []
        for batch in dl:
            torch.cuda.empty_cache()


            x, y, scanner, filepath = batch
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
        #self.gramlayers = [self.stylemodel.layer1[-1].conv1,
        #                   self.stylemodel.layer2[-1].conv1,
        #                   self.stylemodel.layer3[-1].conv1,
        #                   self.stylemodel.layer4[-1].conv1]
        self.gramlayers = [self.stylemodel.down_tr64.ops[1].conv1]
        self.register_hooks()
        logging.info('Gram hooks and memory initialized. Cachesize: %i' % self.hparams.memorymaximum)

    def gram_matrix_3d(self, input):
        # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        a, b, c, d, e = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        grams = []

        for i in range(a):
            features = input[i].view(b, c * d * e)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product
            grams.append(G.div(b * c * d * e))

        return grams

    def gram_hook(self, m, input, output):
        self.grammatrices.append(self.gram_matrix_3d(input[0]))

    def register_hooks(self):
        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

    def training_step(self, batch, batch_idx):
        x, y, filepath, scanner = batch

        self.budget += len(filepath) * self.budgetrate

        #TODO: insert checkpoints for BWT/FWT calculation here

        if ('1.5T Philips' in scanner) and ('3.0T Philips' in scanner):  # this is not the most elegant thing to do
            if not self.shiftcheckpoint_1:
                exp_name = utils.get_expname(self.hparams)
                weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_1_ckpt.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.shiftcheckpoint_1 = True
        elif ('3.0T Philips' in scanner) and ('3.0T' in scanner):
            if not self.shiftcheckpoint_2:
                exp_name = utils.get_expname(self.hparams)
                weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_2_ckpt.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.shiftcheckpoint_2 = True

        if self.hparams.use_memory:
            torch.cuda.empty_cache()

            y = y[:, None]
            self.grammatrices = []
            y_style = self.stylemodel(x.float())

            budget_before = self.budget

            for i, img in enumerate(x):
                grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]

                new_mi = MemoryItem(img, y[i], filepath[i], scanner[i], grammatrix[0])
                self.budget = self.trainingsmemory.insert_element(new_mi, self.budget)

            self.budget = self.trainingsmemory.check_outlier_memory(self.budget)
            self.trainingsmemory.counter_outlier_memory()

            #form trainings X domain balanced batches to train one epoch on all newly inserted samples
            #print(self.trainingsmemory.domaincomplete.items())
            if not np.all(list(self.trainingsmemory.domaincomplete.values())) and budget_before!=self.budget: #only train when a domain is incomplete and new samples are inserted?
                self.eval()
                for k, v in self.trainingsmemory.domaincomplete.items():
                    if not v:
                        domainitems = self.trainingsmemory.get_domainitems(k)
                        if len(domainitems) >= self.trainingsmemory.max_per_domain:
                            preds = []
                            true = []
                            for item in domainitems:
                                x = item.img
                                x = x[None, :].to(self.device)
                                true.append(item.label)
                                outy = self.model(x.float())
                                preds.append(outy.detach().cpu().numpy()[0])
                                # mae = self.mae(y, outy)
                                # error += mae.item()

                            error = mean_absolute_error(true, preds)
                            if error <= self.hparams.completion_limit:
                                self.trainingsmemory.domaincomplete[k] = True

                self.train()

                xs, ys = self.trainingsmemory.get_training_batch(self.hparams.batch_size, batches=2)

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
        #return torch.optim.Adam(self.parameters(), lr=0.00005)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



    #@pl.data_loader
    def train_dataloader(self):
        if self.hparams.continuous:
            return DataLoader(BrainAgeContinuous(self.hparams.datasetfile,
                                                               transition_phase_after=self.hparams.transition_phase_after),
                              batch_size=self.hparams.batch_size, num_workers=4, drop_last=True, pin_memory=False)
        else:
            return DataLoader(BrainAgeDataset(self.hparams.datasetfile,
                                              iterations=self.hparams.noncontinous_steps,
                                              batch_size=self.hparams.batch_size,
                                              split=self.hparams.noncontinous_train_splits),
                              batch_size=self.hparams.batch_size, num_workers=4, pin_memory=False)

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(BrainAgeDataset(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=2, pin_memory=False, drop_last=True)

class DynamicMemoryAge():

    def __init__(self, initelements, memorymaximum=256, gram_weights=None, seed=None):
        self.memoryfull = False
        self.memorylist = initelements
        self.memorymaximum = memorymaximum
        self.samples_per_domain = memorymaximum
        self.gram_weigths = gram_weights
        self.domaincounter = {0: len(self.memorylist)} #0 is the base training domain
        self.max_per_domain = memorymaximum
        self.seed = seed

        graminits = []
        for mi in initelements:
            graminits.append(mi.current_grammatrix)

        self.transformer = SparseRandomProjection(random_state=seed, n_components=30)
        self.transformer.fit(graminits)

        for mi in initelements:
            mi.current_grammatrix = self.transformer.transform(mi.current_grammatrix.reshape(1, -1))

        trans_initelements = self.transformer.transform(graminits)
        clf = IsolationForest(n_estimators=10, random_state=seed).fit(trans_initelements)
        self.isoforests = {0: clf}

        self.domaincomplete = {0: True}

        self.outlier_memory = []
        self.outlier_epochs = 25

        self.img_size = (64, 128, 128)

    def check_outlier_memory(self, budget):
        if len(self.outlier_memory)>10 and int(budget)>=5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]
            # TODO: have to do a pre selection of cache elements here based on, shouldnt add a new pseudodomain if memory is to far spread
            # Add up all pairwise distances if median distance smaller than threshold insert new domain.

            distances = squareform(pdist(outlier_grams)) #distances to see the compactness of memory
            #print(sorted([np.array(sorted(d)[:6]).sum() for d in distances]),
            #      sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5])


            if sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5]<0.15:

                clf = IsolationForest(n_estimators=5, random_state=self.seed, warm_start=True, contamination=0.10).fit(
                    outlier_grams)

                new_domain_label = len(self.isoforests)
                self.domaincomplete[new_domain_label] = False
                self.domaincounter[new_domain_label] = 0
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
                    else:
                        print('run out of budget ', budget)
                for elem in to_delete:
                    self.outlier_memory.remove(elem)

                self.isoforests[new_domain_label] = clf

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

    def insert_element(self, item, budget):
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
                else:
                    self.domaincounter[domain] += 1
                self.memorylist[idx] = item

                # add tree to clf of domain
                clf = self.isoforests[domain]
                domain_items = self.get_domainitems(domain)
                domain_grams = [d.current_grammatrix for d in domain_items]

                if len(clf.estimators_) < 10:
                    n_estimators = len(clf.estimators_) + 1
                    clf.__setattr__('n_estimators', n_estimators)
                else:
                    clf = IsolationForest(n_estimators=10, random_state=16131345)

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

            bs -= j
            if bs>0:
                random.shuffle(self.memorylist)
                for mi in self.memorylist[-bs:]:
                    x[j] = mi.img
                    y[j] = mi.label
                    j += 1

            xs.append(x)
            ys.append(y)

        return (xs, ys)

    def get_domainitems(self, domain):
        items = []
        for mi in self.memorylist:
            if mi.pseudo_domain == domain:
                items.append(mi)
        return items

class MemoryItem():

    def __init__(self, img, label, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.label = label.detach().cpu()
        self.filepath = filepath
        self.scanner = scanner
        self.counter = 0
        self.deleteflag = False
        self.pseudo_domain = pseudo_domain
        self.current_grammatrix = current_grammatrix


def trained_model(hparams):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = FastGramDynamicMemoryBrainAge(hparams=hparams, device=device)
    exp_name = utils.get_expname(model.hparams)
    weights_path = utils.TRAINED_MODELS_FOLDER + exp_name +'.pt'

    if not os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'):
        logger = pllogging.TestTubeLogger(utils.LOGGING_FOLDER, name=exp_name)
        trainer = Trainer(gpus=1, max_epochs=1, logger=logger,
                          val_check_interval=model.hparams.val_check_interval,
                          gradient_clip_val=model.hparams.gradient_clip_val,
                          checkpoint_callback=False)
        trainer.fit(model)
        print('train counter', model.train_counter)
        model.freeze()
        torch.save(model.state_dict(), weights_path)
        if model.hparams.continous and model.hparams.use_memory:
            utils.save_memory_to_csv(model.trainingsmemory.memorylist, utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv')
    else:
        print('Read: ' + weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.freeze()

    if model.hparams.continous and model.hparams.use_memory:
        if os.path.exists(utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv'):
            df_memory = pd.read_csv(utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv')
        else:
            df_memory = None

    # always get the last version
    max_version = max([int(x.split('_')[1]) for x in os.listdir(utils.LOGGING_FOLDER + exp_name)])
    logs = pd.read_csv(utils.LOGGING_FOLDER + exp_name + '/version_{}/metrics.csv'.format(max_version))

    return model, logs, df_memory, exp_name +'.pt'

def is_cached(hparams):
    model = FastGramDynamicMemoryBrainAge(hparams=hparams)
    exp_name = utils.get_expname(model.hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    model = FastGramDynamicMemoryBrainAge(hparams=hparams)
    exp_name = utils.get_expname(model.hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'