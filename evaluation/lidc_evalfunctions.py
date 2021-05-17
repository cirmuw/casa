import torch
from datasets.BatchDataset import LIDCBatch
from datasets.ContinuousDataset import LIDCContinuous
import active_dynamicmemory.LIDCutils as lutils
import active_dynamicmemory.runutils as rutils
import numpy as np
import pandas as pd
import active_dynamicmemory.utils as admutils

def ap_model_hparams(hparams, split='test', scanners=['ges', 'geb', 'sie', 'time_siemens'], dspath='/project/catinous/lungnodulesfinalpatientsplit.csv'):
    device = torch.device('cuda')
    model, _, _, _ = rutils.trained_model(hparams, training=False)
    model.to(device)
    model.eval()
    recalls, precision = ap_model(model, split, scanners=scanners, dspath=dspath)
    return recalls, precision, model


def ap_model(model, split='test', scanners=['ges', 'geb', 'sie', 'time_siemens'], dspath='/project/catinous/lungnodulesfinalpatientsplit.csv'):
    recalls = dict()
    precision = dict()

    step_size = 0.01 #0.05
    for s in scanners:
        recalls[s] = []
        precision[s] = []

    device = torch.device('cuda')

    for res in scanners:
        ds_test = LIDCBatch(dspath,
                            cropped_to=(288, 288), split=split, res=res, validation=True)

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

def recall_precision_to_ap(recalls, precisions, scanners=['ges', 'geb', 'sie', 'time_siemens']):
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

def get_ap_for_res(hparams, split='test', shifts=None, scanners=['ges', 'geb', 'sie', 'time_siemens'], dspath='/project/catinous/lungnodulesfinalpatientsplit.csv'):
    device = torch.device('cuda')
    recalls, precisions, model = ap_model_hparams(hparams, split, scanners=scanners, dspath=dspath)
    aps = recall_precision_to_ap(recalls, precisions, scanners=scanners)
    df_aps = pd.DataFrame([aps])

    if shifts is not None:
        df_aps['shift'] = 'None'

        modelpath = rutils.cached_path(hparams)

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
    return df_aps

def eval_lidc_cont(hparams, seeds=None, split='test', shifts=None, postfixes=None, scanners=['ges', 'geb', 'sie', 'time_siemens'], dspath='/project/catinous/lungnodulesfinalpatientsplit.csv'):
    print('eval for', scanners)
    outputfile = f'/project/catinous/results/lidc/{admutils.get_expname(hparams)}_meanaverageprecision.csv'
    seeds_aps = pd.DataFrame()

    if seeds is not None:
        for i, seed in enumerate(seeds):
            hparams['seed'] = seed
            hparams['run_postfix'] = i+1
            aps = get_ap_for_res(hparams, split=split, shifts=shifts, scanners=scanners, dspath=dspath)
            aps['seed'] = seed
            seeds_aps = seeds_aps.append(aps)
    else:
        for i in range(postfixes):
            hparams['run_postfix'] = i+1
            aps = get_ap_for_res(hparams, split=split, shifts=shifts, scanners=scanners, dspath=dspath)
            seeds_aps = seeds_aps.append(aps)

    seeds_aps.to_csv(outputfile, index=False)