import active_dynamicmemory.FastGramDynamicMemoryBrainAge as braincatsmodel
from active_dynamicmemory.FastGramDynamicMemoryBrainAge import FastGramDynamicMemoryBrainAge
from datasets.BrainAgeDataset import BrainAgeDataset
from datasets.BrainAgeContinuous import BrainAgeContinuous
import active_dynamicmemory.utils as cutils
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import os
import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import sklearn 
from sklearn.metrics import confusion_matrix, auc, roc_curve, mean_absolute_error
import torch
import pandas as pd
import seaborn as sns
import pickle
from py_jotools import slurm, cache
import numpy as np

def test_model(model, ds, device):
    output = []
    target = []
    model.grammatrices = []

    for data in ds:
        x, y, img, res = data
        x = x.float().to(device)
        y_out = model.forward(x)

        target.extend(y.detach().cpu().numpy())
        output.extend([o[0] for o in y_out.detach().cpu().numpy()])
        
        model.grammatrices = []
        
    return target, output


def eval_testset(hparams, dsfile, dssplit, outfile, scanners=['1.5T Philips', '3.0T Philips', '3.0T'], startbudgets=[85, 170, 212, 340], memorymaximas=[128]):
    device = torch.device('cuda')
    
    data_loader = dict()
    for s in scanners:
        data_loader[s] = DataLoader(BrainAgeDataset(dsfile, split=dssplit, res=s), batch_size=2, num_workers=4)
    
    out_scan = []
    out_mm = []
    out_sb = []
    out_post = []
    out_mae = []
    out_method = []
    out_split = []
    for mm in memorymaximas:
        for sb in startbudgets:
            for j in range(3):
                hparams['startbudget'] = sb
                hparams['memorymaximum'] = mm
                hparams['run_postfix'] = j+1
                model, _, _, _ = braincatsmodel.trained_model(hparams, train=False)

                print(f'{mm} {sb} {j}')
                if model is not None:
                    for k in data_loader:
                        target, output = test_model(model, data_loader[k], device)
                        out_scan.append(k)
                        out_mm.append(mm)
                        out_sb.append(sb)
                        out_post.append(j+1)
                        out_mae.append(mean_absolute_error(target, output))
                        out_method.append('casa')
                        out_split.append(dssplit)
                else:
                    print('model is none')

                del model
                torch.cuda.empty_cache()
                print('_________________________________')
    if os.path.exists(outfile):
        df_results = pd.read_csv(outfile)
        df_results = df_results.append(pd.DataFrame({'mm':out_mm, 'sb': out_sb, 'postfix': out_post, 'scanner': out_scan, 'mae': out_mae, 'method': out_method,
                                                     'split': out_split}))
    else:
        df_results = pd.DataFrame({'mm':out_mm, 'sb': out_sb, 'postfix': out_post, 'scanner': out_scan, 'mae': out_mae, 'method': out_method,
                                   'split': out_split})
    df_results.to_csv(outfile, index=False) 
    
def eval_forbwtfwt(hparams, dsfile, dssplit, outfile, method='casa', scanners=['1.5T Philips', '3.0T Philips', '3.0T'], 
                   startbudgets=[85, 170, 212, 340], memorymaximas=[128], continuous_rate = None):
    device = torch.device('cuda')
    
    data_loader = dict()
    for s in scanners:
        data_loader[s] = DataLoader(BrainAgeDataset(dsfile, split=dssplit, res=s), batch_size=2, num_workers=4)
    
    out_scan = []
    out_mm = []
    out_sb = []
    out_post = []
    out_mae = []
    out_method = []
    out_split = []
    out_shift = []
    
    if startbudgets is not None:
        budget_iterator = startbudgets
    else:
        budget_iterator = continuous_rate
    
    for mm in memorymaximas:
        for bi in budget_iterator:
            for j in range(3):
                if startbudgets is not None:
                    hparams['startbudget'] = bi
                else:
                    hparams['naive_continuous_rate'] = bi
                    
                hparams['memorymaximum'] = mm
                hparams['run_postfix'] = j+1
                model, _, _, _ = braincatsmodel.trained_model(hparams, train=False)
                
                print(f'{mm} {bi} {j}')
                if model is not None:
                    
                    exp_name = cutils.get_expname(model.hparams)
                    model.model.load_state_dict(torch.load(cutils.TRAINED_MODELS_FOLDER + exp_name + '_shift_1_ckpt.pt'))
                    model.freeze()
                    
                    for k in data_loader:
                        target, output = test_model(model, data_loader[k], device)
                        out_scan.append(k)
                        out_mm.append(mm)
                        out_sb.append(bi)
                        out_post.append(j+1)
                        out_mae.append(mean_absolute_error(target, output))
                        out_method.append(method)
                        out_split.append(dssplit)
                        out_shift.append(1)
                        
                    exp_name = cutils.get_expname(model.hparams)
                    model.model.load_state_dict(torch.load(cutils.TRAINED_MODELS_FOLDER + exp_name + '_shift_2_ckpt.pt'))
                    model.freeze()
                    
                    for k in data_loader:
                        target, output = test_model(model, data_loader[k], device)
                        out_scan.append(k)
                        out_mm.append(mm)
                        out_sb.append(bi)
                        out_post.append(j+1)
                        out_mae.append(mean_absolute_error(target, output))
                        out_method.append(method)
                        out_split.append(dssplit)
                        out_shift.append(2)
                        
                else:
                    print('model is none')

                del model
                torch.cuda.empty_cache()
                print('_________________________________')
    if os.path.exists(outfile):
        df_results = pd.read_csv(outfile)
        df_results = df_results.append(pd.DataFrame(
            {'mm': out_mm, 'sb': out_sb, 'postfix': out_post, 'scanner': out_scan, 'mae': out_mae, 'method': out_method,
             'shift': out_shift, 'split': out_split}))
    else:
        df_results = pd.DataFrame(
            {'mm': out_mm, 'sb': out_sb, 'postfix': out_post, 'scanner': out_scan, 'mae': out_mae, 'method': out_method,
             'shift': out_shift, 'split': out_split})
    df_results.to_csv(outfile, index=False) 