from pytorch_lightning.utilities.parsing import AttributeDict
import argparse
import pandas as pd
import pytorch_lightning.loggers as pllogging

import torch
import os
import hashlib
import pickle
import skimage.transform
import numpy as np
import pydicom as pyd

import torch
import os
from pytorch_lightning import Trainer
from active_dynamicmemory.CardiacActiveDynamicMemory import CardiacActiveDynamicMemory
from active_dynamicmemory.BrainAgeActiveDynamicMemory import BrainAgeActiveDynamicMemory
from active_dynamicmemory.LIDCActiveDynamicMemory import LIDCActiveDynamicMemory
import pandas as pd
import pytorch_lightning.loggers as pllogging


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
    elif type(hparams) is AttributeDict:
        hparams = dict(hparams)

    hashed_params = hash(hparams, length=10)

    expname = hparams['task']
    expname += '_cont' if hparams['continuous'] else '_batch'

    if 'naive_continuous' in hparams:
        expname += '_naive'

    expname += '_' + os.path.splitext(os.path.basename(hparams['datasetfile']))[0]
    if hparams['base_model']:
        expname += '_basemodel_' + hparams['base_model'].split('_')[1]
    if hparams['continuous']:
        expname += '_memory' if hparams['use_memory'] else '_nomemory'
        expname += '_tf{}'.format(str(hparams['transition_phase_after']).replace('.', ''))
    else:
        expname += '_' + '-'.join(hparams['noncontinuous_train_splits'])
    expname += '_'+str(hparams['run_postfix'])
    expname += '_'+hashed_params
    return expname

def save_memory_to_csv(memory, savepath):
    df_memory = pd.DataFrame({'filepath':[e.filepath for e in memory],
                             'target': [e.target.cpu().numpy()[0] for e in memory],
                             'scanner': [e.scanner for e in memory],
                             'pseudodomain': [e.domain for e in memory]})
    df_memory.to_csv(savepath, index=False, index_label=False)


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
                             'scanner': [ci.scanner for ci in memory],
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

def gram_matrix(input):
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

def gram_matrix_3d(input):
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

def trained_model(hparams, settings, training=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    settings = argparse.Namespace(**settings)
    os.makedirs(settings.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(settings.TRAINED_MEMORY_DIR, exist_ok=True)
    os.makedirs(settings.RESULT_DIR, exist_ok=True)

    if hparams['task'] == 'cardiac':
        model = CardiacActiveDynamicMemory(hparams=hparams, modeldir=settings.TRAINED_MODELS_DIR, device=device, training=training)
    elif hparams['task'] == 'brainage':
        model = BrainAgeActiveDynamicMemory(hparams=hparams, modeldir=settings.TRAINED_MODELS_DIR, device=device, training=training)
    elif hparams['task'] == 'lidc':
        model = LIDCActiveDynamicMemory(hparams=hparams, modeldir=settings.TRAINED_MODELS_DIR, device=device, training=training)
    else:
        raise NotImplementedError('task not implemented')

    exp_name = get_expname(hparams)
    weights_path = cached_path(hparams, settings.TRAINED_MODELS_DIR)

    if not os.path.exists(weights_path) and training:
        logger = pllogging.TestTubeLogger(settings.LOGGING_DIR, name=exp_name)
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
            save_memory_to_csv(model.trainingsmemory.memorylist, settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
    elif os.path.exists(settings.TRAINED_MEMORY_DIR + exp_name + '.pt'):
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
        if os.path.exists(settings.TRAINED_MEMORY_DIR + exp_name + '.csv'):
            df_memory = pd.read_csv(settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
        else:
            df_memory = None
    else:
        df_memory=None

    # always get the last version
    try:
        max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
        logs = pd.read_csv(settings.LOGGING_DIR + exp_name + '/version_{}/metrics.csv'.format(max_version))
    except Exception as e:
        print(e)
        logs = None

    return model, logs, df_memory, exp_name +'.pt'


def is_cached(hparams, trained_dir):
    exp_name = get_expname(hparams)
    return os.path.exists(trained_dir + exp_name + '.pt')


def cached_path(hparams, trained_dir):
    exp_name = get_expname(hparams)
    return trained_dir + exp_name + '.pt'
