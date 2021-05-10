import numpy as np
import SimpleITK as sitk
import torch

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


def correct_boxes(boxes_np, scores_np):
    if len(boxes_np) > 0:
        bidx = torch.ops.torchvision.nms(torch.as_tensor(boxes_np), torch.as_tensor(scores_np), 0.2)

        if len(bidx) == 1:
            final_scores = [np.array(scores_np)[bidx]]
            final_boxes = [np.array(boxes_np)[bidx]]
        else:
            final_scores = np.array(scores_np)[bidx]
            final_boxes = np.array(boxes_np)[bidx]

        return final_boxes, final_scores
    else:
        return boxes_np, scores_np


def load_annotation(self, elem, shiftx_aug=0, shifty_aug=0, ):
    # dcm = pyd.read_file(elem.image, force=True)
    dcm = sitk.ReadImage(elem.image)

    x = elem.x1
    y = elem.y1
    x2 = elem.x2
    y2 = elem.y2

    if self.cropped_to is not None:
        x -= (dcm.GetSize()[0] - self.cropped_to[0]) / 2
        y -= (dcm.GetSize()[1] - self.cropped_to[1]) / 2
        x2 -= (dcm.GetSize()[0] - self.cropped_to[0]) / 2
        y2 -= (dcm.GetSize()[1] - self.cropped_to[1]) / 2

    y -= shiftx_aug
    x -= shifty_aug
    y2 -= shiftx_aug
    x2 -= shifty_aug

    xs = []
    x2s = []
    ys = []
    y2s = []
    for i, row in self.df_multiplenodules.loc[self.df_multiplenodules.image == elem.image].iterrows():

        x1_new = row.x1 - shifty_aug
        x2_new = row.x2 - shifty_aug

        y1_new = row.y1 - shiftx_aug
        y2_new = row.y2 - shiftx_aug

        if x1_new > 0 and x1_new < self.cropped_to[0] and y1_new > 0 and y1_new < self.cropped_to[1]:
            xs.append(x1_new)
            x2s.append(x2_new)

            ys.append(y1_new)
            y2s.append(y2_new)

    if xs == []:
        box = np.zeros((1, 4))
        box[0, 0] = x
        box[0, 1] = y
        box[0, 2] = x2
        box[0, 3] = y2
    else:
        box = np.zeros((len(xs) + 1, 4))
        box[0, 0] = x
        box[0, 1] = y
        box[0, 2] = x2
        box[0, 3] = y2

        for j, x in enumerate(xs):
            box[j + 1, 0] = x
            box[j + 1, 1] = ys[j]
            box[j + 1, 2] = x2s[j]
            box[j + 1, 3] = y2s[j]

    return box

