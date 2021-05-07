import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection
import collections
import numpy as np
from abc import ABC, abstractmethod
import random


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

class DynamicMemory(ABC):

    @abstractmethod
    def __init__(self, initelements, **kwargs):
        self.memoryfull = False
        self.memorylist = initelements
        self.memorymaximum = kwargs['memorymaximum']
        self.seed = kwargs['seed']
        self.labeling_counter = 0

    @abstractmethod
    def insert_element(self, item):
        pass

    def get_training_batch(self, batchsize, force_elements=[], batches=1):
        xs = []
        ys = []

        batchsize = min(batchsize, len(self.memorylist))
        imgshape = self.memorylist[0].img.shape

        half_batch = int(batchsize / 2)

        for b in range(batches):
            j = 0
            bs = batchsize
            if len(imgshape) == 3:
                x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2]))
            elif len(imgshape) == 3:
                x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2], imgshape[3]))

            y = list()

            if len(force_elements) > 0:
                random.shuffle(force_elements)
                m = min(len(force_elements), half_batch)
                for mi in force_elements[-m:]:
                    if j < bs:
                        x[j] = mi.img
                        y.append(mi.target)
                        j += 1
                        mi.traincounter += 1

            bs -= j
            if bs > 0:
                random.shuffle(self.memorylist)
                for mi in self.memorylist[-bs:]:
                    x[j] = mi.img
                    y.append(mi.target)
                    j += 1
                    mi.traincounter += 1

            xs.append(x)
            ys.append(y)
        return xs, ys


class NaiveDynamicMemory(DynamicMemory):

    def __init__(self, initelements, **kwargs):
        super(NaiveDynamicMemory, self).__init__(initelements, **kwargs)
        self.insert_counter = 0
        self.insert_rate = kwargs['insert_rate']
        self.forceitems = []

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


class CasaDynamicMemory(DynamicMemory):

    def __init__(self, initelements, **kwargs):
        super(CasaDynamicMemory, self).__init__(initelements, **kwargs)


        self.samples_per_domain = self.memorymaximum
        self.domaincounter = {0: len(self.memorylist)} #0 is the base training domain
        self.max_per_domain = self.memorymaximum


        graminits = []
        for mi in initelements:
            graminits.append(mi.current_grammatrix)
        print('gram matrix init elements', initelements[0].current_grammatrix.shape)
        if kwargs['transformgrams']:
            self.transformer = SparseRandomProjection(random_state=self.seed, n_components=30)
            self.transformer.fit(graminits)
            print('fit sparse projection')
            for mi in initelements:
                mi.current_grammatrix = self.transformer.transform(mi.current_grammatrix.reshape(1, -1))
            trans_initelements = self.transformer.transform(graminits)
        else:
            self.transformer = None
            trans_initelements = graminits

        clf = IsolationForest(n_estimators=10, random_state=self.seed).fit(trans_initelements)
        self.isoforests = {0: clf}

        self.domaincomplete = {0: True}

        self.perf_queue_len = kwargs['perf_queue_len']
        self.domainMetric = {0: collections.deque(maxlen=self.perf_queue_len)} #TODO: this is an arbritary threshold
        self.outlier_memory = []
        self.outlier_epochs = 15 #TODO: this is an arbritary threshold

        if 'outlier_distance' in kwargs:
            self.outlier_distance = kwargs['outlier_distance']
        else:
            self.outlier_distance = 0.20


    def check_outlier_memory(self, budget, model):
        if len(self.outlier_memory)>5 and int(budget)>=5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]

            distances = squareform(pdist(outlier_grams))
            if sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5]<self.outlier_distance:

                clf = IsolationForest(n_estimators=5, random_state=self.seed, warm_start=True, contamination=0.10).fit(
                    outlier_grams)

                new_domain_label = len(self.isoforests)
                self.domaincomplete[new_domain_label] = False
                self.domaincounter[new_domain_label] = 0
                self.domainMetric[new_domain_label] = collections.deque(maxlen=self.perf_queue_len)
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
                                self.domainMetric[new_domain_label].append(model.get_task_metric(elem.img, elem.target)) #TODO error according to task!
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

        print('detected domain', domain)
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
                self.domainMetric[domain].append(model.get_task_metric(item.img, item.target))

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


    def get_domainitems(self, domain):
        items = []
        for mi in self.memorylist:
            if mi.pseudo_domain == domain:
                items.append(mi)
        return items


class UncertaintyDynamicMemory(DynamicMemory):
    pass

