from datasets.BatchDataset import CardiacBatch
from datasets.ContinuousDataset import CardiacContinuous
import torch
from torch.utils.data import DataLoader
import pandas as pd
import yaml
import active_dynamicmemory.utils as admutils
import os
import active_dynamicmemory.runutils as rutils
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from monai.metrics import DiceMetric
import monai.networks.utils as mutils

def eval_cardiac(params, outfile, split='test'):
    """
    Base evaluation for cardiac on image level, stored to an outputfile
    """

    device = torch.device('cuda')

    dl_test = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=16)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    model.to(device)
    model.eval()

    scanners = []
    dice_lv = []
    dice_myo = []
    dice_rv = []
    dice_mean = []
    shifts = []
    img = []

    #final model
    scanners_m, dice_lv_m, dice_myo_m, dice_rv_m, dice_mean_m, img_m = eval_cardiac_dl(model, dl_test)
    scanners.extend(scanners_m)
    dice_lv.extend(dice_lv_m)
    dice_myo.extend(dice_myo_m)
    dice_rv.extend(dice_rv_m)
    dice_mean.extend(dice_mean_m)
    img.extend(img_m)
    shifts.extend(['None']*len(scanners_m))

    modelpath = rutils.cached_path(params['trainparams'], params['settings']['TRAINED_MODELS_DIR'])
    #model shifts
    for s in params['trainparams']['order'][1:]:
        shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
        model.model.load_state_dict(torch.load(shiftmodelpath, map_location=device))
        model.freeze()

        scanners_m, dice_lv_m, dice_myo_m, dice_rv_m, dice_mean_m, img_m = eval_cardiac_dl(model, dl_test)
        scanners.extend(scanners_m)
        dice_lv.extend(dice_lv_m)
        dice_myo.extend(dice_myo_m)
        dice_rv.extend(dice_rv_m)
        dice_mean.extend(dice_mean_m)
        img.extend(img_m)
        shifts.extend([s] * len(scanners_m))

    df_results = pd.DataFrame({'scanner': scanners, 'dice_lv': dice_lv, 'dice_myo': dice_myo, 'dice_rv': dice_rv, 'dice_mean': dice_mean, 'shift': shifts})
    df_results.to_csv(outfile, index=False)

def eval_cardiac_batch(params, outfile, split='test'):
    dl_test = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=16)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    scanners, dice_lv, dice_myo, dice_rv, dice_mean, img =  eval_cardiac_dl(model, dl_test)

    df_results = pd.DataFrame({'scanner': scanners, 'dice_lv': dice_lv, 'dice_myo': dice_myo, 'dice_rv': dice_rv, 'dice_mean': dice_mean})
    df_results.to_csv(outfile, index=False)

def eval_cardiac_dl(model, dl, device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.freeze()
    scanners = []
    dice_lv = []
    dice_myo = []
    dice_rv = []
    dice_mean = []
    img = []

    dm = DiceMetric(include_background=False, reduction='none')

    for batch in dl:
        x, y, scanner, filepath = batch
        x = x.to(device)
        y_hat = model.forward(x)
        y_hat_flat = torch.argmax(y_hat, dim=1)[:, None, :, :]
        y = y.to(device)[:, None, :, :]

        one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
        one_hot_target = mutils.one_hot(y, num_classes=4)

        dice, not_nan = dm(one_hot_pred, one_hot_target)

        for i in range(len(scanner)):
            scanners.append(scanner[i])
            if not_nan[i][0]==1:
                dice_lv.append(float(dice[i][0].detach()))
            else:
                dice_lv.append(None)
            if not_nan[i][1] == 1:
                dice_myo.append(float(dice[i][1].detach()))
            else:
                dice_myo.append(None)
            if not_nan[i][2]==1:
                dice_rv.append(float(dice[i][2].detach()))
            else:
                dice_rv.append(None)

            dice_mean.append(float(dice[i].detach().sum()/not_nan[i].sum()))
            img.append(filepath[i])

    return scanners, dice_lv, dice_myo, dice_rv, dice_mean, img

def eval_params(params, split='test'):
    settings = argparse.Namespace(**params['settings'])

    expname = admutils.get_expname(params['trainparams'])
    order = params['trainparams']['order'].copy()

    if not os.path.exists(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv'):
        if params['trainparams']['continuous'] == False:
            eval_cardiac_batch(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv', split=split)
        else:
            eval_cardiac(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv', split=split)

    if params['trainparams']['continuous'] == False:
        df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv')
        df_temp = df.groupby(['scanner']).mean().reset_index()
        return df_temp

    df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv')
    df_temp = df.groupby(['scanner', 'shift']).mean().reset_index()

    df_res = df_temp.loc[df_temp['shift'] == 'None']
    df_bwt_fwt = df_temp.groupby(['scanner', 'shift']).mean().reset_index()
    bwt = {'dice_lv': 0.0, 'dice_myo': 0.0, 'dice_rv': 0.0}
    fwt = {'dice_lv': 0.0, 'dice_myo': 0.0, 'dice_rv': 0.0}

    for i in range(len(order) - 1):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i]]
        bwt['dice_lv'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_lv.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_lv.values[0]
        bwt['dice_myo'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_myo.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_myo.values[0]
        bwt['dice_rv'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_rv.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_rv.values[0]

    order.append('None')

    for i in range(2, len(order)):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i - 1]]
        fwt['dice_lv'] += df_scanner.loc[df_scanner['shift'] == order[i]].dice_lv.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[1]].dice_lv.values[0]
        fwt['dice_myo'] += df_scanner.loc[df_scanner['shift'] == order[i]].dice_myo.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[1]].dice_myo.values[0]
        fwt['dice_rv'] += df_scanner.loc[df_scanner['shift'] == order[i]].dice_rv.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[1]].dice_rv.values[0]

    bwt['dice_lv'] /= len(order) - 1
    bwt['dice_myo'] /= len(order) - 1
    bwt['dice_rv'] /= len(order) - 1

    # bwt['mean'] = (bwt['dice_lv']+bwt['dice_myo']+bwt['dice_rv'])/3

    fwt['dice_lv'] /= len(order) - 1
    fwt['dice_myo'] /= len(order) - 1
    fwt['dice_rv'] /= len(order) - 1
    df_res = df_res.append(pd.DataFrame({'scanner': ['BWT', 'FWT'], 'shift': ['None', 'None'],
                                         'dice_lv': [bwt['dice_lv'], fwt['dice_lv']],
                                         'dice_myo': [bwt['dice_myo'], fwt['dice_myo']],
                                         'dice_rv': [bwt['dice_rv'], fwt['dice_rv']]}))

    df_res['mean'] = df_res.mean(axis=1)

    return df_res

def eval_config(configfile, seeds=None, name=None, split='test'):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if seeds is None:
        df = eval_params(params, split=split)
        if name is not None:
            df['model'] = name
        return df
    else:
        df = pd.DataFrame()
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = seed
            df_temp = eval_params(params, split=split)
            df_temp['seed'] = seed
            if name is not None:
                df_temp['model'] = name
            df = df.append(df_temp)

        return df

def eval_config_list(configfiles, names, seeds=None, value='mean'):
    assert type(configfiles) is list, 'configfiles should be a list'
    assert type(names) is list, 'method names should be a list'
    assert len(configfiles) == len(names), 'configfiles and method names should match'

    df_overall = pd.DataFrame()
    for k, configfile in enumerate(configfiles):
        df_conf = eval_config(configfile, seeds, names[k])
        df_overall = df_overall.append(df_conf)

    if value!='std':
        df_overview = df_overall.groupby(['model', 'scanner']).mean().reset_index()
        df_overview = df_overview.pivot(index='model', columns='scanner', values=value).round(3)
    else:
        df_overview = df_overall.groupby(['model', 'scanner']).std().reset_index()
        df_overview = df_overview.pivot(index='model', columns='scanner', values='mean').round(3)

    return df_overview

def val_df_for_params(params):
    exp_name = admutils.get_expname(params['trainparams'])
    settings = argparse.Namespace(**params['settings'])

    max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
    df_temp = pd.read_csv(settings.LOGGING_DIR  + exp_name + '/version_{}/metrics.csv'.format(max_version))

    df_temp = df_temp.loc[df_temp['val_dice_lv_Canon'] == df_temp['val_dice_lv_Canon']]
    df_temp['idx'] = range(1, len(df_temp) + 1)

    return df_temp

def val_data_for_config(configfile, seeds=None):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return val_data_for_params(params, seeds=seeds)

def val_data_for_params(params, seeds=None):
    df = pd.DataFrame()
    if seeds is None:
        df = df.append(val_df_for_params(params))
    else:
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = i + 1
            df = df.append(val_df_for_params(params))

    for scanner in params['trainparams']['order']:
        df[f'val_mean_{scanner}'] = (df[f'val_dice_lv_{scanner}'] + df[f'val_dice_rv_{scanner}'] + df[f'val_dice_myo_{scanner}']) / 3

    return df

def plot_validation_curves(configfiles, val_measure='val_mean', names=None, seeds=None):
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
        ax.set_ylim(0.30, 0.80)
        #ax.set_yticks([0.85, 0.80, 0.75, 0.70])
        ax.get_xaxis().set_visible(False)
        ax.get_legend().remove()
        ax.set_xlim(1, df.idx.max())
        if names is not None:
            ax.set_ylabel(names[k])

        ax.tick_params(labelright=True, right=True)

    # creating timeline
    ds = CardiacContinuous(params['trainparams']['datasetfile'], seed=1)
    res = ds.df.scanner == params['trainparams']['order'][0]
    for j, s in enumerate(params['trainparams']['order'][1:]):
        res[ds.df.scanner == s] = j+2

    axes[-1].imshow(np.tile(res,(400,1)), cmap=ListedColormap(sns.color_palette()[:4]))
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].get_yaxis()