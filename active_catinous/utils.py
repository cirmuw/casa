from py_jotools import mut
import argparse
import pandas as pd
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import os

LOGGING_FOLDER = '/project/catinous/active_catinous/tensorboard_logs/'
TRAINED_MODELS_FOLDER = '/project/catinous/active_catinous/trained_models/'
TRAINED_CACHE_FOLDER = '/project/catinous/active_catinous/trained_cache/'
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

def get_expname(hparams):
    if type(hparams) is argparse.Namespace:
        hparams = vars(hparams).copy()

    hashed_params = mut.hash(hparams, length=10)

    expname = 'cont' if hparams['continuous'] else 'batch'
    expname += '_' + os.path.splitext(os.path.basename(hparams['datasetfile']))
    if hparams['base_model']:
        expname += '_basemodel_' + hparams['base_model'].split('_')[1]
    if hparams['continous']:
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