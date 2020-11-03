from pytorch_lightning.utilities.parsing import AttributeDict
import argparse
import pandas as pd
import pytorch_lightning.loggers as pllogging
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import os
import hashlib
import pickle
import skimage.transform
import numpy as np

LOGGING_FOLDER = '/project/catinous/active_catinous/tensorboard_logs/'
TRAINED_MODELS_FOLDER = '/project/catinous/active_catinous/trained_models/'
TRAINED_MEMORY_FOLDER = '/project/catinous/active_catinous/trained_memory/'
RESPATH = '/project/catinous/active_catinous/results/'

def default_params(dparams, params):
    """Copies all key value pairs from params to dparams if not present"""
    matched_params = dparams.copy()
    default_keys = dparams.keys()
    param_keys = params.keys()
    for key in param_keys:
        matched_params[key] = params[key]
        if key in default_keys:
            if (type(params[key]) is dict) and (type(dparams[key]) is dict):
                matched_params[key] = default_params(dparams[key], params[key])
    return matched_params

def get_expname(hparams, task=None):
    if type(hparams) is argparse.Namespace:
        hparams = vars(hparams).copy()
    elif type(hparams) is AttributeDict:
        hparams = dict(hparams)

    hashed_params = hash(hparams, length=10)

    if task is None:
        expname = 'cont' if hparams['continuous'] else 'batch'
    else:
        expname = task
        expname += '_cont' if hparams['continuous'] else '_batch'

    expname += '_' + os.path.splitext(os.path.basename(hparams['datasetfile']))[0]
    if hparams['base_model']:
        expname += '_basemodel_' + hparams['base_model'].split('_')[1]
    if hparams['continuous']:
        expname += '_fmiss' if hparams['force_misclassified'] else ''
        expname += '_cache' if hparams['use_cache'] else '_nocache'
        expname += '_tf{}'.format(str(hparams['transition_phase_after']).replace('.', ''))
    else:
        expname += '_' + '-'.join(hparams['noncontinous_train_splits'])
    expname += '_'+str(hparams['run_postfix'])
    expname += '_'+hashed_params
    return expname

def save_memory_to_csv(memory, savepath):
    df_memory = pd.DataFrame({'filepath':[e.filepath for e in memory],
                             'target': [e.target.cpu().numpy()[0] for e in memory],
                             'scanner': [e.scanner for e in memory],
                             'pseudodomain': [e.domain for e in memory]})
    df_memory.to_csv(savepath, index=False, index_label=False)

def pllogger(hparams):
    return pllogging.TestTubeLogger(LOGGING_FOLDER, name=get_expname(hparams))

def sort_dict(input_dict):
    dict_out = {}
    keys = list(input_dict.keys())
    keys.sort()
    for key in keys:
        if type(input_dict[key]) is dict:
            value = sort_dict(input_dict[key])
        else:
            value = input_dict[key]
        dict_out[key] = value
    return dict_out


def save_memory_to_csv(memory, savepath):
    df_cache = pd.DataFrame({'filepath':[ci.filepath for ci in memory],
                             'label': [ci.label.cpu().numpy() for ci in memory],
                             'res': [ci.res for ci in memory],
                             'pseudo_domain':  [ci.pseudo_domain for ci in memory],
                             'traincounter': [ci.traincounter for ci in memory]})
    df_cache.to_csv(savepath, index=False, index_label=False)

def hash(item, length=40):
    assert (type(item) is dict)
    item = sort_dict(item)
    return hashlib.sha1(pickle.dumps(item)).hexdigest()[0:length]

def resize(img, size, order=1, anti_aliasing=True):
    for i in range(len(size)):
        if size[i] is None:
            size[i] = img.shape[i]
    return skimage.transform.resize(img, size, order=order, mode='reflect', anti_aliasing=anti_aliasing, preserve_range=True)

def norm01(x):
    """Normalizes values in x to be between 0 and 255"""
    r = (x - np.min(x))
    m = np.max(r)
    if m > 0:
        r = np.divide(r, np.max(r))
    return r