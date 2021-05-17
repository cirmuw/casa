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

def eval_brainage(params, outfile):
    """
    Base evaluation for brainage on image level, stored to an outputfile
    """

    device = torch.device('cuda')

    dl_test = DataLoader(BrainAgeBatch(params['trainparams']['datasetfile'], split=['test']), batch_size=8)
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


def eval_brainage_batch(params, outfile):
    dl_test = DataLoader(BrainAgeBatch(params['trainparams']['datasetfile'], split=['test']), batch_size=8)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    scanners, aes, img =  eval_brainage_dl(model, dl_test)

    df_results = pd.DataFrame({'scanner': scanners, 'AE': aes})
    df_results.to_csv(outfile, index=False)

def eval_brainage_dl(model, dl, device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.eval()

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



def eval_params(params):
    settings = argparse.Namespace(**params['settings'])

    expname = admutils.get_expname(params['trainparams'])
    order = params['trainparams']['order']

    if not os.path.exists(f'{settings.RESULT_DIR}/cache/{expname}_AEs.csv'):
        if params['trainparams']['continuous'] == False:
            eval_brainage_batch(params, f'{settings.RESULT_DIR}/cache/{expname}_AEs.csv')
        else:
            eval_brainage(params, f'{settings.RESULT_DIR}/cache/{expname}_AEs.csv')

    if params['trainparams']['continuous'] == False:
        df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_AEs.csv')
        df_temp = df.groupby(['scanner']).mean().reset_index()
        return df_temp

    df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_AEs.csv')
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
    fwt['dice_rv'] /= len(order) - 1
    df_res = df_res.append(pd.DataFrame({'scanner': ['BWT', 'FWT'], 'shift': ['None', 'None'],
                                         'ae': [bwt, fwt]}))

    return df_res


def eval_config(configfile, seeds=None, name=None):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if seeds is None:
        df = eval_params(params)
        if name is not None:
            df['model'] = name
        return df
    else:
        df = pd.DataFrame()
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = i+1
            df_temp = eval_params(params)
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

    df_overview = df_overall.groupby(['model', 'scanner']).mean().reset_index()
    df_overview = df_overview.pivot(index='model', columns='scanner', values=value).round(3)

    return df_overview