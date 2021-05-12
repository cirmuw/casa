import torch
import torch.nn as nn
from datasets.ContinuousDataset import LIDCContinuous
from datasets.BatchDataset import LIDCBatch
from active_dynamicmemory.ActiveDynamicMemoryModel import ActiveDynamicMemoryModel
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import active_dynamicmemory.LIDCutils as lutils
import numpy as np

class LIDCActiveDynamicMemory(ActiveDynamicMemoryModel):

    def __init__(self, hparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        super(ActiveDynamicMemoryModel, self).__init__()
        self.TaskDatasetBatch = LIDCBatch
        self.TaskDatasetContinuous = LIDCContinuous

        self.collate_fn = lutils.collate_fn

        self.init(hparams=hparams, modeldir=modeldir, device=device, training=training)


    def get_task_metric(self, image, target):
        """
        Task metric for LIDC is the mean intersection over union
        :param image: image the absolute error should be calculted for
        :param target: real age of the patient
        :return: mean intersection over union
        """
        self.eval()

        out = self.model(image[None, :, :].to(self.device))

        out_boxes = [
            lutils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(), out[i]['scores'].cpu().detach().numpy())
            for i in range(len(out))]

        boxes_np = [b[0] for b in out_boxes]
        scores_np = [b[1] for b in out_boxes]

        final_boxes = []
        final_scores = []
        for i, box_np in enumerate(boxes_np):
            fb, fs = lutils.correct_boxes(box_np, scores_np[i])
            final_boxes.append(fb)
            final_scores.append(fs)

        ious = []
        for t in target:
            for b in final_boxes[0]:
                ious.append(lutils.bb_intersection_over_union(t, b))
        self.train()
        return np.array(ious).mean()


    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        """
        Load the cardiac segmentation model (Res. U-Net)
        :param droprate: dropout rate to be applied
        :param load_stylemodel: If true loads the style model (needed for training)
        :return: loaded model, stylemodel and gramlayers
        """
        num_classes = 3  # 0=background, 1=begnin, 2=malignant
        # load a model pre-trained pre-trained on COCO
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if droprate!=0.0:
            model.backbone.body.layer1 = lutils.add_dropout_to_seq(model.backbone.body.layer1)
            model.backbone.body.layer2 = lutils.add_dropout_to_seq(model.backbone.body.layer2)
            model.backbone.body.layer3 = lutils.add_dropout_to_seq(model.backbone.body.layer3)
            model.backbone.body.layer4 = lutils.add_dropout_to_seq(model.backbone.body.layer4)

        if load_stylemodel:
            stylemodel = models.resnet50(pretrained=True)
            gramlayers = [stylemodel.layer1[-1].conv1]
            stylemodel.eval()

            return model, stylemodel, gramlayers

        return model, None, None


    def completed_domain(self, m):
        """
        Domain is completed if m smaller than a threshold
        :param m: value to compare to the threshold
        :return: Wheter or not the domain is considered completed
        """
        return m>self.hparams.completion_limit

    def validation_step(self, batch, batch_idx):
        self.grammatrices = []
        images, targets, scanner, _ = batch
        images = list(image.to(self.device) for image in images)

        out = self.model(images)

        out_boxes = [
            lutils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(), out[i]['scores'].cpu().detach().numpy())
            for i in range(len(out))]

        boxes_np = [b[0] for b in out_boxes]
        scores_np = [b[1] for b in out_boxes]

        final_boxes = []
        final_scores = []

        for i, box_np in enumerate(boxes_np):
            fb, fs = lutils.correct_boxes(box_np, scores_np[i])
            final_boxes.append(fb)
            final_scores.append(fs)

        gt = []
        for t in targets:
            gt.append(t['boxes'])

        return {'final_boxes': final_boxes, 'final_scores': final_scores, 'gt': gt, 'scanner': scanner}

    def validation_epoch_end(self, validation_step_outputs):
        iou_thres = 0.2

        overall_true_pos = dict()
        overall_false_pos = dict()
        overall_false_neg = dict()
        overall_boxes_count = dict()
        recalls = dict()
        precision = dict()

        for scanner in self.hparams.order:
            overall_true_pos[scanner] = dict()
            overall_false_pos[scanner] = dict()
            overall_false_neg[scanner] = dict()
            overall_boxes_count[scanner] = dict()
            recalls[scanner] = []
            precision[scanner] = []
            for k in np.arange(0.0, 1.01, 0.05):
                overall_true_pos[scanner][k] = 0
                overall_false_pos[scanner][k] = 0
                overall_false_neg[scanner][k] = 0
                overall_boxes_count[scanner][k] = 0

        for out in validation_step_outputs:
            final_boxes = out['final_boxes']
            final_scores = out['final_scores']
            gt = out['gt']
            scanner = out['scanner']

            for j, fb in enumerate(final_boxes):
                s = scanner[j]
                g = gt[j]
                fs = final_scores[j]

                for k in np.arange(0.0, 1.01, 0.05):
                    false_positives = 0
                    false_negatives = 0
                    true_positives = 0
                    detected = [False] * len(g)
                    boxes_count = 0
                    if len(fb) > 0:
                        for i, b in enumerate(fb):
                            if fs[i] > k:
                                boxes_count += 1
                                det_gt = False
                                for m, singleg in enumerate(g):
                                    if lutils.bb_intersection_over_union(singleg, b) > iou_thres:
                                        detected[m] = True
                                        det_gt = True
                                if not det_gt:
                                    false_positives += 1
                    for d in detected:
                        if d:
                            true_positives += 1
                        else:
                            false_negatives += 1
                    overall_true_pos[s][k] += true_positives
                    overall_false_pos[s][k] += false_positives
                    overall_false_neg[s][k] += false_negatives
                    overall_boxes_count[s][k] += boxes_count

        aps = dict()

        for scanner in self.hparams.order:
            for k in np.arange(0.0, 1.01, 0.05):
                if (overall_false_neg[scanner][k] + overall_true_pos[scanner][k]) == 0:
                    recalls[scanner].append(0.0)
                else:
                    recalls[scanner].append(
                        overall_true_pos[scanner][k] / (overall_false_neg[scanner][k] + overall_true_pos[scanner][k]))
                if (overall_false_pos[scanner][k] + overall_true_pos[scanner][k]) == 0:
                    precision[scanner].append(0.0)
                else:
                    precision[scanner].append(
                        overall_true_pos[scanner][k] / (overall_false_pos[scanner][k] + overall_true_pos[scanner][k]))

            prec = np.array(precision[scanner])
            rec = np.array(recalls[scanner])
            ap = []
            for t in np.arange(0.0, 1.01, 0.1):
                prec_arr = prec[rec > t]
                if len(prec_arr) == 0:
                    ap.append(0.0)
                else:
                    ap.append(prec_arr.max())
            aps[scanner] = np.array(ap).mean()

            self.log(f'val_ap_{scanner}', aps[scanner])

    def get_task_loss(self, xs, ys):
        loss = None

        if type(xs) is list:
            for _xs, _ys in zip(xs, ys):
                x = list(i.to(self.device) for i in _xs)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in _ys]
                loss_dict = self.forward_lidc(x, targets)
                if loss is None:
                    loss = sum(l for l in loss_dict.values())
                else:
                    loss += sum(l for l in loss_dict.values())
        else:
            x = list(i.to(self.device) for i in xs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in ys]
            loss_dict = self.forward_lidc(x, targets)
            loss = sum(l for l in loss_dict.values())

        return loss

    def forward_lidc(self, x, y):
        return self.model(x, y)

    def get_uncertainties(self, x):
        pass