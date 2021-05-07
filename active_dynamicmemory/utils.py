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
        expname += '_fmiss' if hparams['force_misclassified'] else ''
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

def collate_fn(batch):
    return tuple(zip(*batch))

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def filter_boxes_area(boxes, scores, min_area=10):
    out_boxes = []
    out_scores = []
    for i, b in enumerate(boxes):
        area = (b[3] - b[1]) * (b[2] - b[0])
        if area > min_area:
            out_boxes.append(b)
            out_scores.append(scores[i])

    return np.array(out_boxes), np.array(out_scores)


def correct_boxes(boxes_np, scores_np, x_shift=112, y_shift=112):
    if len(boxes_np[0]) > 0:
        boxes_np[0][:, 0] += x_shift
        boxes_np[0][:, 1] += y_shift
        boxes_np[0][:, 2] += x_shift
        boxes_np[0][:, 3] += y_shift

    if len(boxes_np[2]) > 0:
        boxes_np[2][:, 1] += y_shift * 2
        boxes_np[2][:, 3] += y_shift * 2

    if len(boxes_np[3]) > 0:
        boxes_np[3][:, 0] += x_shift * 2
        boxes_np[3][:, 2] += x_shift * 2

    if len(boxes_np[4]) > 0:
        boxes_np[4][:, 0] += x_shift * 2
        boxes_np[4][:, 2] += x_shift * 2
        boxes_np[4][:, 1] += y_shift * 2
        boxes_np[4][:, 3] += y_shift * 2

    # there is a better way for sure... move fast and break things
    final_boxes = []
    final_boxes.extend(boxes_np[0])
    final_boxes.extend(boxes_np[1])
    final_boxes.extend(boxes_np[2])
    final_boxes.extend(boxes_np[3])
    final_boxes.extend(boxes_np[4])

    final_scores = []
    final_scores.extend(scores_np[0])
    final_scores.extend(scores_np[1])
    final_scores.extend(scores_np[2])
    final_scores.extend(scores_np[3])
    final_scores.extend(scores_np[4])

    if len(final_boxes) > 0:
        bidx = torch.ops.torchvision.nms(torch.as_tensor(final_boxes), torch.as_tensor(final_scores), 0.2)

        if len(bidx) == 1:
            final_scores = [np.array(final_scores)[bidx]]
            final_boxes = [np.array(final_boxes)[bidx]]
        else:
            final_scores = np.array(final_scores)[bidx]
            final_boxes = np.array(final_boxes)[bidx]

    return final_boxes, final_scores


def load_box_annotation(elem, cropped_to=None, shiftx_aug=0, shifty_aug=0, validation=False):
    dcm = pyd.read_file(elem.image)
    x = elem.coordX
    y = elem.coordY
    diameter = elem.diameter_mm
    spacing = float(dcm.PixelSpacing[0])

    if not validation:
        if cropped_to is not None:
            x -= (dcm.Rows - cropped_to[0]) / 2
            y -= (dcm.Columns - cropped_to[1]) / 2
        y -= shiftx_aug
        x -= shifty_aug

    x -= int((diameter / spacing) / 2)
    y -= int((diameter / spacing) / 2)

    x2 = x + int(diameter / spacing)
    y2 = y + int(diameter / spacing)

    box = np.zeros((1, 4))
    box[0, 0] = x
    box[0, 1] = y
    box[0, 2] = x2
    box[0, 3] = y2

    return box


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