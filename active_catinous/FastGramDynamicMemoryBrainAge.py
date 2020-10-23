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


from . import utils

SEED = 2314134

class FastGramDynamicMemoryBrainAge():

    def __init__(self, initelements, hparams={}, device=torch.device('cpu'), verbose=False):
        #super(FastGramCacheBrainAge, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)

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

        if self.hparams.use_cache and self.hparams.continous:
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
            self.hparams.use_cache = False

        self.stylemodel.to(device)
        self.stylemodel.eval()

        self.model = EncoderRegressor()
        self.model.to(device)
        if not self.hparams.base_model is None:
            self.load_state_dict(
                torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model), map_location=device))


        self.loss = nn.MSELoss()
        self.mae = nn.L1Loss()

        #PREFILL memory with base training samples!!!!
        self.trainingsmemory = DynamicMemoryAge(initelements=initelements,
                                                cachemaximum=self.hparams.memorymaximum,
                                               balance_cache=self.hparams.balance_memory,
                                               gram_weights=self.hparams.gram_weights)

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
        hparams['misclassified_thresh'] = 4.0
        hparams['order'] = ['1.5T Philips', '3.0T Philips', '3.0T']
        hparams['continuous'] = True
        hparams['noncontinuous_steps'] = 3000
        hparams['noncontinuous_train_splits'] = ['base_train']
        hparams['val_check_interval'] = 100
        hparams['base_model'] = None
        hparams['run_postfix'] = '1'
        hparams['gram_weights'] = [1, 1, 1, 1]

    def init_cache_and_gramhooks(self):
        self.grammatrices = []
        #self.gramlayers = [self.stylemodel.layer1[-1].conv1,
        #                   self.stylemodel.layer2[-1].conv1,
        #                   self.stylemodel.layer3[-1].conv1,
        #                   self.stylemodel.layer4[-1].conv1]
        self.gramlayers = [self.stylemodel.down_tr64.ops[1].conv1]
        self.register_hooks()
        logging.info('Gram hooks and memory initialized. Cachesize: %i' % self.hparams.memorymaximum)

    def gram_matrix(self, input):
        # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        grams = []

        for i in range(a):
            features = input[i].view(b, c * d)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product
            grams.append(G.div(b * c * d))

        return grams

    def gram_hook(self, m, input, output):
        self.grammatrices.append(self.gram_matrix(input[0]))

    def register_hooks(self):
        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

    def training_step(self, batch):
        x, y, filepath, scanner = batch

        #TODO: insert checkpoints for BWT/FWT calculation here


        if self.hparams.use_memory:
            torch.cuda.empty_cache()

            y = y[:, None]
            self.grammatrices = []
            y_style = self.stylemodel(x.float())

            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu() for bg in self.grammatrices]

                new_mi = MemoryItem(img, y[i], filepath[i], scanner[i], grammatrix, transformer=self.trainingsmemory.transformer)
                self.trainingsmemory.insert_element(new_mi)

            self.trainingsmemory.check_outlier_memory()

            self.model.train()
            #form trainings X domain balanced batches to train one epoch on all newly inserted samples
            trainingbatches = self.trainingsmemory.get_training_batch(self.hparams.batchsize)
            if trainingbatches is not None:
                for tb in trainingbatches:
                    x, y = tb

                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x.float())

                    loss = self.loss(y_hat, y.float())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            #TODO:check training for completion?
            self.model.eval()
            for k, v in self.trainingsmemory.domaincomplete.items():
                if not v:
                    #for items in :#TODO: get items with a pseudodomain
                        pass

        def validation(dl_val):
                pass

        def continuous_train(dl, dl_val):
            for i, batch in enumerate(dl):
                self.training_step(batch)

                if (i%10)==0:
                    self.validation(dl_val)

            #check if memory accuracy for domain is reaching a threshold


class DynamicMemoryAge():

    def __init__(self, initelements, memorymaximum=256, gram_weights=None):
        self.memoryfull = False
        self.memorylist = initelements
        self.memorymaximum = memorymaximum
        self.samples_per_domain = memorymaximum
        self.gram_weigths = gram_weights
        self.domaincounter = {0: len(self.memorylist)} #0 is the base training domain
        self.max_per_domain = memorymaximum

        self.transformer = SparseRandomProjection(random_state=SEED, n_components=30)
        self.transformer.fit(initelements)

        trans_initelements = [self.transformer.transform(e) for e in initelements]
        clf = IsolationForest(n_estimators=10, random_state=SEED).fit(trans_initelements)
        self.isoforests = {0: clf}

        self.domaincomplete = {0: True}

        self.outlier_memory = []
        self.outlier_epochs = 10

        self.img_size = (64, 128, 128)

    def check_outlier_memory(self):
        if len(self.outlier_memory)>10:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]
            clf = IsolationForest(n_estimators=5, random_state=SEED, warm_start=True, contamination=0.10).fit(
                outlier_grams) # TODO: have to do a pre selection of cache elements here based on, shouldnt add a new pseudodomain if memory is to far spread

            new_domain_label = len(self.isoforests)
            self.domaincomplete[new_domain_label] = False
            self.domaincounter[new_domain_label] = 0
            self.max_per_domain = int(self.memorymaximum/(new_domain_label+1))

            self.flag_items_for_deletion()

            for k, p in enumerate(clf.predict(outlier_grams)):
                if p == 1:
                    idx = self.find_insert_position()
                    if idx != -1:
                        self.memorylist.append(self.outlier_memory[k])
                        self.outlier_memory.remove(self.outlier_memory[k])
                        self.domaincounter[new_domain_label] += 1

            self.isoforests[new_domain_label] = clf

    def find_insert_position(self):
        for idx, item in enumerate(self.memorylist):
            if item.deleteflag:
                return idx
        return -1

    def flag_items_for_deletion(self):
        #TODO find a smart way to do this
        for k, v in self.domaincounter.items():
            if v>self.max_per_domain:
                todelete = v-self.max_per_domain
                for item in self.memorylist:
                    if not item.deleteflag:
                        item.deleteflag = True
                    else:
                        todelete -= 1


    def counter_outlier_memory(self):
        for item in self.outlier_memory:
            item.counter += 1
            if item.counter>self.outlier_epochs:
                self.outlier_memory.remove(item)

    def insert_element(self, item):
        domain = self.check_pseudodomain(item.current_grammatrix)
        item.pseudo_domain = domain

        if domain==-1:
            #insert into outlier memory
            #check outlier memory for new clusters
            self.outlier_memory.append(item)
        else:
            if not self.domaincomplete[domain]:
                #insert into dynamic memory and training
                idx = self.find_insert_position()
                if idx == -1: # no free memory position, replace an element already in memory
                    mingramloss = 1000
                    for j, mi in enumerate(self.memorylist):
                        if mi.pseudodomain == domain:
                            loss = F.mse_loss(item.current_grammatrix, mi.current_grammatrix, reduction='mean')

                            if loss < mingramloss:
                                mingramloss = loss
                                idx = j
                self.memorylist[idx] = item
                self.domaincounter[domain] += 1


    def check_pseudodomain(self, grammatrix):
        max_pred = 0
        current_domain = -1

        for j, clf in enumerate(self.isoforests):
            current_pred = clf.decision_function(grammatrix.reshape(1, -1))

            if current_pred>max_pred:
                current_domain = j
                max_pred = current_pred

        return current_domain

    def get_training_batch(self, batchsize): #TODO: force new items!!
        if np.all(list(self.domain_complete.values())):
            return None
        else:
            j = 0
            x = torch.empty(size=(batchsize, 1, self.img_size[0], self.img_size[1], self.img_size[2]))
            y = torch.empty(size=(batchsize, 1))

            elements_per_domain = round(batchsize/len(self.domaincounter))

            for d in self.domaincounter:
                domain_items = []
                for mi in self.memorylist:
                    if mi.pseudo_domain == d:
                        domain_items.append(mi)

                random.shuffle(domain_items)
                for mi in domain_items[-elements_per_domain:]:
                    x[j] = mi.img
                    y[j] = mi.label
                    j += 1

            batchsize -= j
            if batchsize>0:
                random.shuffle(domain_items)
                for mi in domain_items[-batchsize:]:
                    x[j] = mi.img
                    y[j] = mi.label
                    j += 1

            return x, y

class MemoryItem():

    def __init__(self, img, label, filepath, scanner, current_grammatrix=None, transformer=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.label = label.detach().cpu()
        self.filepath = filepath
        self.scanner = scanner
        self.counter = 0
        self.deleteflag = False
        self.pseudo_domain = pseudo_domain

        if transformer is not None:
            current_grammatrix = transformer.transform(current_grammatrix)

        self.current_grammatrix = current_grammatrix
