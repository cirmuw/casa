import torch
from datasets.BatchDataset import LIDCBatch
from datasets.ContinuousDataset import LIDCContinuous
import active_dynamicmemory.LIDCutils as lutils
import active_dynamicmemory.runutils as rutils
import numpy as np
import pandas as pd
import active_dynamicmemory.utils as admutils
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def ap_model_mparams(mparams, split='test', scanners=['ges', 'geb', 'sie', 'lndb'], dspath='/project/catinous/lungnodulesfinallndbBig.csv'):
    device = torch.device('cuda')
    model, _, _, _ = rutils.trained_model(mparams['trainparams'], mparams['settings'], training=False)
    model.to(device)
    model.eval()
    recalls, precision = ap_model(model, split, scanners=scanners, dspath=dspath)
    return recalls, precision, model


def ap_model(model, split='test', scanners=['ges', 'geb', 'sie', 'lndb'], dspath='/project/catinous/lungnodulesfinallndbBig.csv'):
    recalls = dict()
    precision = dict()

    step_size = 0.01 #0.05
    for s in scanners:
        recalls[s] = []
        precision[s] = []

    device = torch.device('cuda')

    for res in scanners:
        ds_test = LIDCBatch(dspath, split=split, res=res, validation=True)

        iou_thres = 0.2

        overall_true_pos = dict()
        overall_false_pos = dict()
        overall_false_neg = dict()
        overall_boxes_count = dict()
        for k in np.arange(0.0, 1.01, step_size):
            overall_true_pos[k] = 0
            overall_false_pos[k] = 0
            overall_false_neg[k] = 0
            overall_boxes_count[k] = 0

        for batch in ds_test:
            img_batch, annot, res, image = batch
            img_batch = img_batch[None, :, :, :]
            img_batch = img_batch.to(device)

            out = model.model(img_batch)
            out_boxes = [lutils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(),
                                                  out[i]['scores'].cpu().detach().numpy()) for i in range(len(out))]
            boxes_np = [b[0] for b in out_boxes]
            scores_np = [b[1] for b in out_boxes]

            final_boxes, final_scores = lutils.correct_boxes(boxes_np[0], scores_np[0])

            gt = annot['boxes']
            #if res=='time_siemens':
            #    print('new time series')
            #    print(gt)
            #    print(final_boxes, len(final_boxes))
            for k in np.arange(0.0, 1.01, step_size):
                false_positives = 0
                false_negatives = 0
                true_positives = 0
                detected = [False]*len(gt)
                boxes_count = 0
                if len(final_boxes) > 0:
                    for i, b in enumerate(final_boxes):
                        if final_scores[i] > k:
                            boxes_count += 1
                            detected_gt = False
                            for j, g in enumerate(gt):
                                #if res=='time_siemens':
                                    #print(cutils.bb_intersection_over_union(g, b), 'intersect')
                                if lutils.bb_intersection_over_union(g, b) > iou_thres:
                                    detected[j] = True
                                    detected_gt = True
                            if not detected_gt:
                                false_positives += 1
                for d in detected:
                    if d:
                        true_positives+=1
                    else:
                        false_negatives+=1

                overall_true_pos[k] += true_positives
                overall_false_pos[k] += false_positives
                overall_false_neg[k] += false_negatives
                overall_boxes_count[k] += boxes_count
        for k in np.arange(0.0, 1.01, step_size):
            if (overall_false_neg[k] + overall_true_pos[k]) == 0:
                recalls[res].append(0.0)
            else:
                recalls[res].append(overall_true_pos[k] / (overall_false_neg[k] + overall_true_pos[k]))
            if (overall_false_pos[k] + overall_true_pos[k]) == 0:
                precision[res].append(0.0)
            else:
                precision[res].append(overall_true_pos[k] / (overall_false_pos[k] + overall_true_pos[k]))
    return recalls, precision

def recall_precision_to_ap(recalls, precisions, scanners=['ges', 'geb', 'sie', 'lndb']):
    aps = dict()
    for res in scanners:
        prec = np.array(precisions[res])
        rec = np.array(recalls[res])
        ap = []
        for t in np.arange(0.0, 1.01, 0.1):
            prec_arr = prec[rec > t]
            if len(prec_arr) == 0:
                ap.append(0.0)
            else:
                ap.append(prec_arr.max())
        aps[res] = np.array(ap).mean()
    return aps

def get_ap_for_res(params, split='test', shifts=None, scanners=['ges', 'geb', 'sie', 'lndb'], dspath='/project/catinous/lungnodulesfinallndbBig.csv', force_recalculation = False):
    device = torch.device('cuda')
    expname = admutils.get_expname(params['trainparams'])
    settings = argparse.Namespace(**params['settings'])


    if not os.path.exists(f'{settings.RESULT_DIR}/cache/{expname}_{split}_aps.csv') or force_recalculation:
        recalls, precisions, model = ap_model_mparams(params, split, scanners=scanners, dspath=dspath)
        aps = recall_precision_to_ap(recalls, precisions, scanners=scanners)
        df_aps = pd.DataFrame([aps])

        if shifts is not None:
            df_aps['shift'] = 'None'

            modelpath = rutils.cached_path(params['trainparams'], params['settings']['TRAINED_MODELS_DIR'])

            for s in shifts:
                shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
                print('starting load shift model', s, shiftmodelpath)
                model.model.load_state_dict(torch.load(shiftmodelpath, map_location=device))
                model.freeze()
                print('starting to eval on shiftmodel', s)

                recalls, precisions = ap_model(model, split, scanners=scanners, dspath=dspath)
                aps = recall_precision_to_ap(recalls, precisions, scanners=scanners)
                aps = pd.DataFrame([aps])
                aps['shift'] = s
                df_aps = df_aps.append(aps)
        df_aps.to_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_aps.csv')
    else:
        df_aps = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_aps.csv')

    return df_aps

def eval_lidc_cont(params, seeds=None, split='test', shifts=None, scanners=['ges', 'geb', 'sie', 'lndb'], dspath='/project/catinous/lungnodulesfinallndbBig.csv'):
    print('eval for', scanners)
    #outputfile = f'/project/catinous/results/lidc/{admutils.get_expname(mparams)}_meanaverageprecision.csv'
    seeds_aps = pd.DataFrame()

    if seeds is not None:
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = i+1
            aps = get_ap_for_res(params, split=split, shifts=shifts, scanners=scanners, dspath=dspath)
            aps['seed'] = seed
            seeds_aps = seeds_aps.append(aps)
    else:
        aps = get_ap_for_res(params, split=split, shifts=shifts, scanners=scanners, dspath=dspath)
        seeds_aps = seeds_aps.append(aps)

    #seeds_aps.to_csv(outputfile, index=False)

    return seeds_aps
    
def val_data_for_config(configfile, seeds=None):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return eval_lidc_cont(params, seeds=seeds)


def eval_config(configfile, shifts=None, seeds=None, name=None, split='test', force_recalculation = False):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if seeds is None:
        df = get_ap_for_res(params, shifts=shifts, split=split, force_recalculation = force_recalculation)
        if name is not None:
            df['model'] = name
        return df
    else:
        df = pd.DataFrame()
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = seed
            df_temp = get_ap_for_res(params, shifts=shifts, split=split, force_recalculation = force_recalculation)

            df_res = df_temp.loc[df_temp['shift'] == 'None']

            bwt = 0.0
            fwt = 0.0
            order = ['ges', 'geb', 'sie', 'lndb']

            for i in range(len(order) - 1):
                bwt += df_temp.loc[df_temp['shift'] == 'None'][order[i]].values[0] - \
                       df_temp.loc[df_temp['shift'] == order[i + 1]][order[i]].values[0]

            bwt /= len(order) - 1

            order.append('None')
            for i in range(2, len(order)):
                fwt += df_temp.loc[df_temp['shift'] == order[i]][order[i - 1]].values[0] - \
                       df_temp.loc[df_temp['shift'] == order[1]][order[i - 1]].values[0]

            fwt /= len(order) - 1

            df_res['bwt'] = bwt
            df_res['fwt'] = fwt

            df_res['seed'] = seed
            if name is not None:
                df_res['model'] = name
            df = df.append(df_res)

        return df

def eval_config_list(configfiles, names, seeds=None, split='test', value='mean'):
    assert type(configfiles) is list, 'configfiles should be a list'
    assert type(names) is list, 'method names should be a list'
    assert len(configfiles) == len(names), 'configfiles and method names should match'

    df_overall = pd.DataFrame()
    for k, configfile in enumerate(configfiles):
        df_conf = eval_config(configfile, seeds, names[k], split=split)
        df_overall = df_overall.append(df_conf)

    if value=='mean':
        df_overview = df_overall.groupby(['model']).mean().reset_index()
    elif value=='std':
        df_overview = df_overall.groupby(['model']).std().reset_index()

    return df_overview

def val_data_for_params(params, seeds=None):
    df = pd.DataFrame()
    if seeds is None:
        df = df.append(val_df_for_params(params))
    else:
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = i + 1
            df = df.append(val_df_for_params(params))

    return df

def val_df_for_params(params):
    exp_name = admutils.get_expname(params['trainparams'])
    settings = argparse.Namespace(**params['settings'])

    max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
    df_temp = pd.read_csv(settings.LOGGING_DIR  + exp_name + '/version_{}/metrics.csv'.format(max_version))

    df_temp = df_temp.loc[df_temp['val_ap_geb'] == df_temp['val_ap_geb']]
    df_temp['idx'] = range(1, len(df_temp) + 1)

    return df_temp

def plot_validation_curves(configfiles, val_measure='val_ap', names=None, seeds=None):
    assert type(configfiles) is list, "configfiles should be a list"

    fig, axes = plt.subplots(len(configfiles)+1, 1, figsize=(10, 2.5*(len(configfiles))))
    plt.subplots_adjust(hspace=0.0)

    for k, configfile in enumerate(configfiles):
        with open(configfile) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        df = val_data_for_params(params, seeds=seeds)
        ax = axes[k]
        for scanner in params['trainparams']['order']:
            sns.lineplot(data=df, y=f'{val_measure}_{scanner}', x='idx', ax=ax, label=scanner)
        #ax.set_ylim(0.30, 0.80)
        #ax.set_yticks([0.85, 0.80, 0.75, 0.70])
        ax.get_xaxis().set_visible(False)
        ax.get_legend().remove()
        ax.set_xlim(1, df.idx.max())
        if names is not None:
            ax.set_ylabel(names[k])

        ax.tick_params(labelright=True, right=True)

    # creating timeline
    ds = LIDCContinuous(params['trainparams']['datasetfile'], seed=1)
    res = ds.df.scanner == params['trainparams']['order'][0]
    for j, s in enumerate(params['trainparams']['order'][1:]):
        res[ds.df.scanner == s] = j+2

    axes[-1].imshow(np.tile(res,(400,1)), cmap=ListedColormap(sns.color_palette()[:4]))
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].get_yaxis()