import math
import torch
from datasets.BatchDataset import BrainAgeBatch
import pandas as pd
import active_dynamicmemory.runutils as rutils
from torch.utils.data import DataLoader
import argparse
import active_dynamicmemory.utils as admutils
import os
import yaml

def eval_brainage(params, outfile, split=['test']):
    """
    Base evaluation for brainage on image level, stored to an outputfile
    """

    device = torch.device('cuda')

    dl_test = DataLoader(BrainAgeBatch(params['trainparams']['datasetfile'], split=split), batch_size=8)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    model.to(device)
    model.eval()

    scanners = []
    aes = []
    shifts = []
    img = []

    #final model
    scanners_m, aes_m, img_m = eval_brainage_dl(model, dl_test)
    scanners.extend(scanners_m)
    aes.extend(aes_m)
    img.extend(img_m)
    shifts.extend(['None']*len(scanners_m))

    modelpath = rutils.cached_path(params['trainparams'], params['settings']['TRAINED_MODELS_DIR'])
    #model shifts
    for s in params['trainparams']['order'][1:]:
        shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
        model.model.load_state_dict(torch.load(shiftmodelpath, map_location=device))
        model.freeze()

        scanners_m, aes_m, img_m = eval_brainage_dl(model, dl_test)
        scanners.extend(scanners_m)
        aes.extend(aes_m)
        img.extend(img_m)
        shifts.extend([s] * len(scanners_m))

    df_results = pd.DataFrame({'scanner': scanners, 'ae': aes, 'shift': shifts})
    df_results.to_csv(outfile, index=False)


def eval_brainage_batch(params, outfile, split=['test']):
    dl_test = DataLoader(BrainAgeBatch(params['trainparams']['datasetfile'], split=split), batch_size=8)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    scanners, aes, img =  eval_brainage_dl(model, dl_test)

    df_results = pd.DataFrame({'scanner': scanners, 'AE': aes})
    df_results.to_csv(outfile, index=False)

def eval_brainage_dl(model, dl, device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.freeze()

    scanners = []
    img = []
    aes = []

    for batch in dl:
        x, y, scanner, filepath = batch
        x = x.to(device)
        y_hat = model.forward(x)

        for i, m in enumerate(y):
            aes.append(float(abs(y_hat[i]-m)[0]))
        scanners.extend(scanner)
        img.extend(filepath)

    return scanners, aes, img



def eval_params(params, split='test'):
    settings = argparse.Namespace(**params['settings'])
    expname = admutils.get_expname(params['trainparams'])
    order = params['trainparams']['order'].copy()

    if not os.path.exists(f'{settings.RESULT_DIR}/cache/{expname}_{split}_AEs.csv'):
        if params['trainparams']['continuous'] == False:
            eval_brainage_batch(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_AEs.csv', split=split)
        else:
            eval_brainage(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_AEs.csv', split=split)

    if params['trainparams']['continuous'] == False:
        df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_AEs.csv')
        df_temp = df.groupby(['scanner']).mean().reset_index()
        return df_temp

    df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_AEs.csv')
    df_temp = df.groupby(['scanner', 'shift']).mean().reset_index()

    df_res = df_temp.loc[df_temp['shift'] == 'None']
    df_bwt_fwt = df_temp.groupby(['scanner', 'shift']).mean().reset_index()
    bwt = 0.0
    fwt = 0.0

    for i in range(len(order) - 1):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i]]
        bwt = df_scanner.loc[df_scanner['shift'] == 'None'].ae.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[i + 1]].ae.values[0]

    order.append('None')

    for i in range(2, len(order)):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i - 1]]
        fwt += df_scanner.loc[df_scanner['shift'] == order[i]].ae.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[1]].ae.values[0]

    bwt /= len(order) - 1
    fwt /= len(order) - 1
    df_res = df_res.append(pd.DataFrame({'scanner': ['BWT', 'FWT'], 'shift': ['None', 'None'],
                                         'ae': [bwt, fwt]}))

    return df_res


def eval_config(configfile, seeds=None, name=None, split=['test']):
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
            params['trainparams']['run_postfix'] = i+1
            df_temp = eval_params(params, split=split)
            df_temp['seed'] = seed
            if name is not None:
                df_temp['model'] = name
            df = df.append(df_temp)

        return df

def eval_config_list(configfiles, names, seeds=None, value='mean', split=['test']):
    assert type(configfiles) is list, 'configfiles should be a list'
    assert type(names) is list, 'method names should be a list'
    assert len(configfiles) == len(names), 'configfiles and method names should match'

    df_overall = pd.DataFrame()
    for k, configfile in enumerate(configfiles):
        df_conf = eval_config(configfile, seeds, names[k], split=split)
        df_overall = df_overall.append(df_conf)

    df_overview = df_overall.groupby(['model', 'scanner']).mean().reset_index()
    df_overview = df_overview.pivot(index='model', columns='scanner', values=value).round(3)

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

    print(df_temp)
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