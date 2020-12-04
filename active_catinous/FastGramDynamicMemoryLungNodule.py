import argparse
import logging
import os
import random
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pllogging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

from datasets.LIDCDatasetContinuous import LIDCDatasetContinuous
from datasets.LIDCDataset import LIDCDataset

from sklearn.metrics import mean_absolute_error

from scipy.spatial.distance import pdist, squareform
import torchvision.models as tvmodels
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import utils
import collections
import copy
import pickle

class FastGramDynamicMemoryLungNodule(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbose=False, training=True):
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
        num_classes = 3  # 0=background, 1=begnin, 2=malignant
        # load a model pre-trained pre-trained on COCO
        self.model = tvmodels.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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
            print('only validation, not training the model')
        elif self.hparams.continuous:
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
                                                                seed=self.hparams.seed)
                else:
                    print('init CASA memory')
                    self.trainingsmemory = DynamicMemoryLN(initelements=initmemoryelements,
                                                           memorymaximum=self.hparams.memorymaximum,
                                                           seed=self.hparams.seed,
                                                           perf_queue_len=self.hparams.len_perf_queue)

                    print(len(self.trainingsmemory.get_domainitems(0)))
            else:
                self.hparams.use_memory = False

    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['datasetfile'] = '/project/catinous/lunadata/luna_lunacombined_dataset_malignancy.csv'
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
        hparams['iou_threshold'] = 0.2

        return hparams

    def getmemoryitems_from_base(self, num_items=128):
        dl = DataLoader(LIDCDataset(self.hparams.datasetfile,
                                   iterations=None,
                                   batch_size=self.hparams.batch_size,
                                   split=['base_train']),
                   batch_size=4, num_workers=4, pin_memory=True, shuffle=True,
                        collate_fn=utils.collate_fn)

        memoryitems = []
        for batch in dl:
            self.grammatrices = []
            torch.cuda.empty_cache()

            images, targets, scanner, filepath = batch
            images = torch.stack(images)

            x = images.to(self.device)
            y_style = self.stylemodel(x)

            for i, f in enumerate(filepath):
                memoryitems.append(MemoryItem(x[i].detach().cpu(), targets[i], f, scanner[i],
                                              current_grammatrix=self.grammatrices[0][i].detach().cpu().numpy().flatten(),
                                              pseudo_domain=0))

            if len(memoryitems)>=num_items:
                break

            self.grammatrices = []

        with open('initimages.pkl', 'wb') as f:
            pickle.dump(images, f)

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
        images, targets, scanner, filepath = batch

        self.budget += len(filepath) * self.budgetrate

        if self.hparams.order[1] in scanner and not self.shiftcheckpoint_1:
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_1_ckpt.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.shiftcheckpoint_1 = True
        if self.hparams.order[2] in scanner and not self.shiftcheckpoint_2:
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_2_ckpt.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.shiftcheckpoint_2 = True

        if not self.naive_continuous and self.hparams.use_memory:
                torch.cuda.empty_cache()
                self.grammatrices = []
                #images = list(image.to(self.device) for image in images)
                img_tensors = torch.stack(images).to(self.device)
                y_style = self.stylemodel(img_tensors)

                budget_before = self.budget

                if batch_idx==0:
                    with open('grammatrices.pkl', 'wb') as f:
                        pickle.dump(self.grammatrices, f)
                    with open('batchimages.pkl', 'wb') as f:
                        pickle.dump(img_tensors, f)

                for i, img in enumerate(images):
                    new_mi = MemoryItem(img.detach().cpu(), targets[i], filepath[i], scanner[i], self.grammatrices[0][i].detach().cpu().numpy().flatten())
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
                            if len(self.trainingsmemory.domainPerf[k])==self.hparams.len_perf_queue:
                                meanperf = np.mean(self.trainingsmemory.domainPerf[k])
                                print('domain', k, meanperf, self.trainingsmemory.domainPerf[k])
                                if meanperf>self.hparams.completion_limit: #TODO < or > depending on metric
                                    self.trainingsmemory.domaincomplete[k] = True
                                    print('domain', k, 'finished')

                    images, targets = self.trainingsmemory.get_training_batch(self.hparams.batch_size)

                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    loss_dict = self.forward(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    self.train_counter += 1
                    self.log_dict(loss_dict)
                    self.log('train_loss', losses)
                    return losses
                else:
                    return None
        elif self.naive_continuous and self.hparams.use_memory:
            torch.cuda.empty_cache()
            self.grammatrices = []

            images = list(image.to(self.device) for image in images)
            y_style = self.stylemodel(images)

            for i, img in enumerate(images):
                grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
                new_mi = MemoryItem(img.detach().cpu(), targets[i], filepath[i], scanner[i], grammatrix[0])
                self.trainingsmemory.insert_element(new_mi)

            if len(self.trainingsmemory.forceitems)!=0:
                images, targets = self.trainingsmemory.get_training_batch(self.hparams.batch_size)

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.forward(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.train_counter += 1
                self.log_dict(loss_dict)
                self.log('train_loss', losses)
                return losses
            else:
                return None
        else:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.forward(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.log_dict(loss_dict)
            self.log('train_loss', losses)
            return losses

    def validation_step(self, batch, batch_idx):
        images, targets, scanner, filepath = batch
        images = list(image.to(self.device) for image in images[0])

        out = self.model(images)

        out_boxes = [utils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(), out[i]['scores'].cpu().detach().numpy())
                     for i in range(len(out))]
        boxes_np = [b[0] for b in out_boxes]
        scores_np = [b[1] for b in out_boxes]

        final_boxes, final_scores = utils.correct_boxes(boxes_np, scores_np)

        gt = targets[0]['boxes'][0]
        true_pred = 0
        wrong_pred = 0
        detected = False

        if len(final_boxes) > 0:
            for i, b in enumerate(final_boxes):
                if final_scores[i] > 0.5:
                    if utils.bb_intersection_over_union(gt, b) > self.hparams.iou_threshold:
                        true_pred += 1
                        detected = True
                    else:
                        wrong_pred += 1

        return {'true': true_pred, 'false': wrong_pred, 'detected': detected, 'scanner': scanner[0]}

    def validation_epoch_end(self, validation_step_outputs):
        scanner_tp = dict()
        scanner_fp = dict()
        scanner_det = dict()

        for scanner in self.hparams.order:
            scanner_tp[scanner] = []
            scanner_fp[scanner] = []
            scanner_det[scanner] = []
        for pred in validation_step_outputs:
            tp = pred['true']
            fp = pred['false']
            det = pred['detected']
            scanner = pred['scanner']
            scanner_tp[scanner].append(tp)
            scanner_fp[scanner].append(fp)
            scanner_det[scanner].append(det)


        print(scanner_tp, scanner_fp, scanner_det)
        for scanner in scanner_tp:
            true = np.array(scanner_tp[scanner]).sum()
            false = np.array(scanner_fp[scanner]).sum()

            print(true/(true+false), np.array(scanner_det[scanner]).sum()/len(scanner_det[scanner]), false/len(scanner_det[scanner]))
            self.log(f'val_map_{scanner}', true/(true+false))
            self.log(f'val_mapdet_{scanner}', np.array(scanner_det[scanner]).sum()/len(scanner_det[scanner]))
            self.log(f'val_fpscan_{scanner}', false/len(scanner_det[scanner]))

    def forward(self, x, y):
        return self.model(x, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def check_detection(self, img, target):
        img = [img.to(self.device)]

        self.model.eval()
        out = self.model(img)
        self.model.train()

        out_boxes = out[0]['boxes'].cpu().detach().numpy()
        out_scores = out[0]['scores'].cpu().detach().numpy()
        gt = target['boxes'][0]

        detected = 0
        for i, b in enumerate(out_boxes):
            if out_scores[i] > 0.5:
                if utils.bb_intersection_over_union(gt, b) > self.hparams.iou_threshold:
                    detected = 1

        return detected


    #@pl.data_loader
    def train_dataloader(self):
        if self.hparams.continuous:
            return DataLoader(LIDCDatasetContinuous(self.hparams.datasetfile,
                                                               transition_phase_after=self.hparams.transition_phase_after,
                                                                seed=self.hparams.seed),
                              batch_size=self.hparams.batch_size, num_workers=4, drop_last=True, pin_memory=False, collate_fn=utils.collate_fn)
        else:
            return DataLoader(LIDCDataset(self.hparams.datasetfile,
                                              iterations=self.hparams.noncontinuous_steps,
                                              batch_size=self.hparams.batch_size,
                                              split=self.hparams.noncontinuous_train_splits,
                                              res=self.hparams.scanner,
                                              seed=self.hparams.seed,
                                            cropped_to=(288, 288)
                                              ),
                              batch_size=self.hparams.batch_size, num_workers=4, pin_memory=False,
                              collate_fn=utils.collate_fn)

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(LIDCDataset(self.hparams.datasetfile,
                                          split='val', validation=True, cropped_to=(288, 288)),
                          batch_size=1,
                          num_workers=2, pin_memory=False, drop_last=False,
                          collate_fn=utils.collate_fn)

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
            with open('graminits.pkl', 'wb') as f:
                pickle.dump(graminits, f)
            for mi in initelements:
                mi.current_grammatrix = self.transformer.transform(mi.current_grammatrix.reshape(1, -1))[0]
            trans_initelements = self.transformer.transform(graminits)
        else:
            self.transformer = None
            trans_initelements = graminits

        clf = IsolationForest(n_estimators=10, random_state=seed).fit(trans_initelements)
        self.isoforests = {0: clf}

        self.domaincomplete = {0: True}

        self.domainPerf = {0: collections.deque(maxlen=perf_queue_len)}
        self.perf_queue_len = perf_queue_len
        self.outlier_memory = []
        self.outlier_epochs = 10 #TODO: this is an arbritary threshold

        self.img_size = (288, 288)

        print('init memory item', self.memorylist[0].current_grammatrix)

    def check_outlier_memory(self, budget, model):
        if len(self.outlier_memory)>5 and int(budget)>=5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]

            distances = squareform(pdist(outlier_grams))
            print('check outlier memory distances', distances)
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
                                self.domainPerf[new_domain_label].append(model.check_detection(elem.img, elem.target))
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
        print('domain detected', domain)
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
                self.domainPerf[domain].append(model.check_detection(item.img, item.target))

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

    def get_cropped_mi(self, mi, cropped_to):
        img = mi.img
        crop_target = copy.deepcopy(mi.target)
        box = crop_target['boxes']

        if box[0, 3] > cropped_to[0]:
            min_x = box[0, 3] - cropped_to[0]
            max_x = img.shape[1] - cropped_to[0]
        else:
            min_x = 0
            max_x = cropped_to[0] - box[0, 3]

        if box[0, 2] > cropped_to[1]:
            min_y = box[0, 2] - cropped_to[1]
            max_y = img.shape[2] - cropped_to[1]
        else:
            min_y = 0
            max_y = cropped_to[1] - box[0, 2]

        start_x = random.randint(min_x, max_x)
        start_y = random.randint(min_y, max_y)

        crop_img = img[:, start_x:start_x + cropped_to[0], start_y:start_y + cropped_to[1]]
        box[0, 0] -= start_y
        box[0, 1] -= start_x
        box[0, 2] -= start_y
        box[0, 3] -= start_x

        crop_target['boxes'] = box


        return crop_img, crop_target

    def get_training_batch(self, batchsize, cropped_to=(288, 288)):
        xs = []
        ys = []

        to_force = []

        half_batch = int(batchsize/2)

        for d, c in self.domaincomplete.items():
            if not c:
                for mi in self.memorylist:
                    if mi.pseudo_domain == d:
                        to_force.append(mi)

        j = 0
        bs = batchsize
        images = []
        targets = []

        random.shuffle(to_force)
        for mi in to_force[-half_batch:]:
            if j<bs:
                crop_img, crop_target = self.get_cropped_mi(mi, cropped_to)
                images.append(crop_img)
                targets.append(crop_target)
                j += 1
                mi.traincounter += 1

        bs -= j
        if bs>0:
            random.shuffle(self.memorylist)
            for mi in self.memorylist[-bs:]:
                crop_img, crop_target = self.get_cropped_mi(mi, cropped_to)
                images.append(crop_img)
                targets.append(crop_target)
                j += 1
                mi.traincounter += 1

        return (images, targets)

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

    def get_cropped_mi(self, mi, cropped_to):
        img = mi.img
        crop_target = copy.deepcopy(mi.target)
        box = crop_target['boxes']

        if box[0, 3] > cropped_to[0]:
            min_x = box[0, 3] - cropped_to[0]
            max_x = img.shape[1] - cropped_to[0]
        else:
            min_x = 0
            max_x = cropped_to[0] - box[0, 3]

        if box[0, 2] > cropped_to[1]:
            min_y = box[0, 2] - cropped_to[1]
            max_y = img.shape[2] - cropped_to[1]
        else:
            min_y = 0
            max_y = cropped_to[1] - box[0, 2]

        start_x = random.randint(min_x, max_x)
        start_y = random.randint(min_y, max_y)

        crop_img = img[:, start_x:start_x + cropped_to[0], start_y:start_y + cropped_to[1]]
        box[0, 0] -= start_y
        box[0, 1] -= start_x
        box[0, 2] -= start_y
        box[0, 3] -= start_x

        crop_target['boxes'] = box


        return crop_img, crop_target

    def get_training_batch(self, batchsize, cropped_to=(288, 288)):
        xs = []
        ys = []

        half_batch = int(batchsize / 2)

        j = 0
        bs = batchsize
        images = []
        targets = []

        if len(self.forceitems)>0:
            random.shuffle(self.forceitems)
            m = min(len(self.forceitems), half_batch)
            for mi in self.forceitems[-m:]:
                if j < bs:
                    crop_img, crop_target = self.get_cropped_mi(mi, cropped_to)
                    images.append(crop_img)
                    targets.append(crop_target)
                    j += 1
                    mi.traincounter += 1

        bs -= j
        if bs>0:
            random.shuffle(self.memorylist)
            for mi in self.memorylist[-bs:]:
                crop_img, crop_target = self.get_cropped_mi(mi, cropped_to)
                images.append(crop_img)
                targets.append(crop_target)
                j += 1
                mi.traincounter += 1


        self.forceitems = []

        return (images, targets)


class MemoryItem():

    def __init__(self, img, target, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.target = target
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
    model = FastGramDynamicMemoryLungNodule(hparams=hparams, device=device, training=train)
    exp_name = utils.get_expname(model.hparams, task='ln')
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
    hparams = utils.default_params(FastGramDynamicMemoryLungNodule.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams, task='ln')
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    #model = FastGramDynamicMemoryBrainAge(hparams=hparams)
    hparams = utils.default_params(FastGramDynamicMemoryLungNodule.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams, task='ln')
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'