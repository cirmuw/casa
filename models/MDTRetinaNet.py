import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import nms

## Taken from https://github.com/MIC-DKFZ/medicaldetectiontoolkit/

#File: retina_net.py

class config():
    def __init__(self):
        self.dim = 2
        self.pre_crop_size = [300, 300]
        self.patch_size = [288, 288]
        self.head_classes = 2
        self.start_filts = 48
        self.end_filts = self.start_filts * 4
        self.n_rpn_features = 512
        self.rpn_anchor_ratios = [0.5, 1, 2]
        self.rpn_anchor_stride = 1
        self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3
        self.relu = 'relu' #alternativ: leaky_relu
        self.pre_nms_limit = 3000
        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1]])
        self.model_max_instances_per_batch_element = 10  # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1
        self.rpn_anchor_scales = {'xy': [[8], [16], [32], [64]], 'z': [[2], [4], [8], [16]]}
        self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}
        self.pyramid_levels = [0, 1, 2, 3]

        self.backbone_shapes = np.array(
            [[int(np.ceil(self.patch_size[0] / stride)),
              int(np.ceil(self.patch_size[1] / stride))]
             for stride in self.backbone_strides['xy']])
        
        self.n_channels = 1 #can be set to 3 to include prev and succ slice
        self.norm = None


############################################################
#  Network Heads
############################################################

class Classifier(nn.Module):

    def __init__(self, cf, conv):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.n_classes = cf.head_classes
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * cf.head_classes
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)

        return [class_logits]


class BBRegressor(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * self.dim * 2
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride,
                               pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        return [bb_logits]


############################################################
#  Output Handler
############################################################

def refine_detections(anchors, probs, deltas, batch_ixs, cf):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    anchors = anchors.repeat(len(np.unique(batch_ixs)), 1)

    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:cf.pre_nms_limit]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)

    pre_nms_scores = flat_probs[:cf.pre_nms_limit]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).long().cuda()

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = apply_box_deltas_2D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = clip_to_window(cf.window, refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]

            xywh = [[ab[0], ab[1], ab[2] - ab[0], ab[3] - ab[1]] for ab in ix_rois.cpu().detach().numpy()]
            class_keep = nms.boxes(xywh, ix_scores, nms_threshold=cf.detection_nms_threshold)

            #if cf.dim == 2:
            #    class_keep = nms_2D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), cf.detection_nms_threshold)
            #else:
            #    class_keep = nms_3D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), cf.detection_nms_threshold)

            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result


def get_results(cf, img_shape, detections, seg_logits, box_results_list=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                          retina_unet and dummy array for retina_net.
    """
    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, cf.dim*2]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:

            boxes = detections[ix][:, :2 * cf.dim].astype(np.int32)
            class_ids = detections[ix][:, 2 * cf.dim + 1].astype(np.int32)
            scores = detections[ix][:, 2 * cf.dim + 2]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            if cf.dim == 2:
                exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            else:
                exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)

            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    if score >= cf.model_min_confidence:
                        box_results_list[ix].append({'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_type': 'det',
                                                     'box_pred_class_id': class_ids[ix2]})

    results_dict = {'boxes': box_results_list}
    if seg_logits is None:
        # output dummy segmentation for retina_net.
        results_dict['seg_preds'] = np.zeros(img_shape)[:, 0][:, np.newaxis]
    else:
        # output label maps for retina_unet.
        results_dict['seg_preds'] = F.softmax(seg_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis].astype('uint8')

    return results_dict



class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts = cf.start_filts
        start_filts = self.start_filts
        self.n_blocks = [3, 4, 6, 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = False#cf.sixth_pooling
        self.dim = conv.dim

        if operate_stride1:
            self.C0 = nn.Sequential(conv(cf.n_channels, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu),
                                    conv(start_filts, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu))

            self.C1 = conv(start_filts, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        else:
            self.C1 = conv(cf.n_channels, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                         if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(
                start_filts_exp * 8, start_filts * 16, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 8, 2, 2)))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(start_filts_exp * 16, start_filts * 16, conv=conv, norm=cf.norm, relu=cf.relu))
            self.C6 = nn.Sequential(*C6_layers)

        if conv.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        self.out_channels = cf.end_filts
        self.P5_conv1 = conv(start_filts*32, self.out_channels, ks=1, stride=1, relu=None) #
        self.P4_conv1 = conv(start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
        #     print ("encoder shapes:", ii.shape)
        #
        # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
        #     print("decoder shapes:", ii.shape)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list



class ResBlock(nn.Module):

    def __init__(self, start_filts, planes, conv, stride=1, downsample=None, norm=None, relu='relu'):
        super(ResBlock, self).__init__()
        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, planes * 4, ks=1, norm=norm, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], norm=norm, relu=None)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


############################################################
#  Retina (U-)Net Class
############################################################


class net(nn.Module):


    def __init__(self, cf, logger):

        super(net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.build()

    def build(self):
        """
        Build Retina Net architecture.
        """

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # instanciate abstract multi dimensional conv class and backbone model.
        conv = NDConvGenerator(self.cf.dim)

        # build Anchors, FPN, Classifier / Bbox-Regressor -head
        self.np_anchors = generate_pyramid_anchors(self.logger, self.cf)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()
        self.Fpn = FPN(self.cf, conv)
        self.Classifier = Classifier(self.cf, conv)
        self.BBRegressor = BBRegressor(self.cf, conv)


    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        gt_class_ids = batch['roi_labels']
        gt_boxes = batch['bb_target']

        img = torch.from_numpy(img).float().cuda()
        batch_class_loss = torch.FloatTensor([0]).cuda()
        batch_bbox_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        detections, class_logits, pred_deltas, seg_logits = self.forward(img)

        # loop over batch
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': batch['bb_target'][b][ix],
                                                'box_label': batch['roi_labels'][b][ix], 'box_type': 'gt'})

                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas = gt_anchor_matching(
                    self.cf, self.np_anchors, gt_boxes[b], gt_class_ids[b])

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

            # compute losses.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

            # add negative anchors used for loss to results_dict for monitoring.
            neg_anchors = clip_boxes_numpy(
                self.np_anchors[np.argwhere(anchor_class_match == -1)][0, neg_anchor_ix], img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss / img.shape[0]

        results_dict = get_results(self.cf, img.shape, detections, seg_logits, box_results_list)
        loss = batch_class_loss + batch_bbox_loss
        results_dict['torch_loss'] = loss
        results_dict['monitor_values'] = {'loss': loss.item(), 'class_loss': batch_class_loss.item()}
        results_dict['logger_string'] = "loss: {0:.2f}, class: {1:.2f}, bbox: {2:.2f}"\
            .format(loss.item(), batch_class_loss.item(), batch_bbox_loss.item())

        return results_dict


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                            retina_unet and dummy array for retina_net.
        """
        img = batch['data']
        #img = torch.from_numpy(img).float().cuda()
        detections, _, _, seg_logits = self.forward(img)
        results_dict = get_results(self.cf, img.shape, detections, seg_logits)
        return results_dict


    def forward(self, img):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # Feature extraction
        fpn_outs = self.Fpn(img)
        seg_logits = None
        selected_fmaps = [fpn_outs[i] for i in self.cf.pyramid_levels]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.Classifier(p))
            bb_reg_layer_outputs.append(self.BBRegressor(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]

        # merge batch_dimension and store info in batch_ixs for re-allocation.
        batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).cuda()
        flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
        detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, self.cf)

        return detections, class_logits, bb_outputs, seg_logits

############################################################
#  Loss Functions
############################################################

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.size()[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        # return the indices of negative samples, which contributed to the loss (for monitoring plots).
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

#model_utils.py

def shem(roi_probs_neg, negative_count, ohem_poolsize):
    """
    stochastic hard example mining: from a list of indices (referring to non-matched predictions),
    determine a pool of highest scoring (worst false positives) of size negative_count*ohem_poolsize.
    Then, sample n (= negative_count) predictions of this pool as negative examples for loss.
    :param roi_probs_neg: tensor of shape (n_predictions, n_classes).
    :param negative_count: int.
    :param ohem_poolsize: int.
    :return: (negative_count).  indices refer to the positions in roi_probs_neg. If pool smaller than expected due to
    limited negative proposals availabel, this function will return sampled indices of number < negative_count without
    throwing an error.
    """
    # sort according to higehst foreground score.
    probs, order = roi_probs_neg[:, 1:].max(1)[0].sort(descending=True)
    select = torch.tensor((ohem_poolsize * int(negative_count), order.size()[0])).min().int()
    pool_indices = order[:select]
    rand_idx = torch.randperm(pool_indices.size()[0])
    return pool_indices[rand_idx[:negative_count].cuda()]

def apply_box_deltas_2D(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2) / 3D: (z1, z2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]  / 3D: (z1, z2)
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    if boxes.shape[1] > 5:
        boxes[:, 4] = boxes[:, 4].clamp(float(window[4]), float(window[5]))
        boxes[:, 5] = boxes[:, 5].clamp(float(window[4]), float(window[5]))

    return boxes

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = torch.ByteTensor([True], requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]


class NDConvGenerator(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                conv = nn.Sequential(conv, norm_layer)

        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)

        return conv

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def generate_pyramid_anchors(logger, cf):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    from configs:
    :param scales: cf.RPN_ANCHOR_SCALES , e.g. [4, 8, 16, 32]
    :param ratios: cf.RPN_ANCHOR_RATIOS , e.g. [0.5, 1, 2]
    :param feature_shapes: cf.BACKBONE_SHAPES , e.g.  [array of shapes per feature map] [80, 40, 20, 10, 5]
    :param feature_strides: cf.BACKBONE_STRIDES , e.g. [2, 4, 8, 16, 32, 64]
    :param anchors_stride: cf.RPN_ANCHOR_STRIDE , e.g. 1
    :return anchors: (N, (y1, x1, y2, x2, (z1), (z2)). All generated anchors in one array. Sorted
    with the same order of the given scales. So, anchors of scale[0] come first, then anchors of scale[1], and so on.
    """
    scales = cf.rpn_anchor_scales
    ratios = cf.rpn_anchor_ratios
    feature_shapes = cf.backbone_shapes
    anchor_stride = cf.rpn_anchor_stride
    pyramid_levels = cf.pyramid_levels
    feature_strides = cf.backbone_strides

    anchors = []
    logger.info("feature map shapes: {}".format(feature_shapes))
    logger.info("anchor scales: {}".format(scales))

    expected_anchors = [np.prod(feature_shapes[ii]) * len(ratios) * len(scales['xy'][ii]) for ii in pyramid_levels]

    for lix, level in enumerate(pyramid_levels):
        anchors.append(generate_anchors(scales['xy'][level], ratios, feature_shapes[level],
                                        feature_strides['xy'][level], anchor_stride))
        logger.info("level {}: built anchors {} / expected anchors {} ||| total build {} / total expected {}".format(
            level, anchors[-1].shape, expected_anchors[lix], np.concatenate(anchors).shape, np.sum(expected_anchors)))

    out_anchors = np.concatenate(anchors, axis=0)
    return out_anchors



def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i] #this is the gt box
        overlaps[:, i] = compute_iou_2D(box2, boxes1, area2[i], area1)
    return overlaps


def compute_iou_2D(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2] THIS IS THE GT BOX
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union

    return iou

def gt_anchor_matching(cf, anchors, gt_boxes, gt_class_ids=None):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, (z1), (z2))]
    gt_class_ids (optional): [num_gt_boxes] Integer class IDs for one stage detectors. in RPN case of Mask R-CNN,
    set all positive matches to 1 (foreground)
    Returns:
    anchor_class_matches: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral.
               In case of one stage detectors like RetinaNet/RetinaUNet this flag takes
               class_ids as positive anchor values, i.e. values >= 1!
    anchor_delta_targets: [N, (dy, dx, (dz), log(dh), log(dw), (log(dd)))] Anchor bbox deltas.
    """

    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((cf.rpn_train_anchors_per_image, 2*cf.dim))
    anchor_matching_iou = cf.anchor_matching_iou

    if gt_boxes is None:
        anchor_class_matches = np.full(anchor_class_matches.shape, fill_value=-1)
        return anchor_class_matches, anchor_delta_targets

    # for mrcnn: anchor matching is done for RPN loss, so positive labels are all 1 (foreground)
    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= anchor_matching_iou then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.1 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.1).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    if anchors.shape[1] == 4:
        anchor_class_matches[(anchor_iou_max < 0.1)] = -1
    elif anchors.shape[1] == 6:
        anchor_class_matches[(anchor_iou_max < 0.01)] = -1
    else:
        raise ValueError('anchor shape wrong {}'.format(anchors.shape))

    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    for ix, ii in enumerate(gt_iou_argmax):
        anchor_class_matches[ii] = gt_class_ids[ix]

    # 3. Set anchors with high overlap as positive.
    above_trhesh_ixs = np.argwhere(anchor_iou_max >= anchor_matching_iou)
    anchor_class_matches[above_trhesh_ixs] = gt_class_ids[anchor_iou_argmax[above_trhesh_ixs]]

    # Subsample to balance positive anchors.
    ids = np.where(anchor_class_matches > 0)[0]
    extra = len(ids) - (cf.rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[ids] = 0

    # Leave all negative proposals negative now and sample from them in online hard example mining.
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(anchor_class_matches > 0)[0]
    ix = 0  # index into anchor_delta_targets
    for i, a in zip(ids, anchors[ids]):
        # closest gt box (it might have IoU < anchor_matching_iou)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # convert coordinates to center plus width/height.
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        if cf.dim == 2:
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]

        else:
            gt_d = gt[5] - gt[4]
            gt_center_z = gt[4] + 0.5 * gt_d
            a_d = a[5] - a[4]
            a_center_z = a[4] + 0.5 * a_d

            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                (gt_center_z - a_center_z) / a_d,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
                np.log(gt_d / a_d)
            ]

        # normalize.
        anchor_delta_targets[ix] /= cf.rpn_bbox_std_dev
        ix += 1

    return anchor_class_matches, anchor_delta_targets

def clip_boxes_numpy(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2 / [N, 6] in 3D.
    window: iamge shape (y, x, (z))
    """
    if boxes.shape[1] == 4:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
            np.clip(boxes[:, 1], 0, window[0])[:, None],
            np.clip(boxes[:, 2], 0, window[1])[:, None],
            np.clip(boxes[:, 3], 0, window[1])[:, None]), 1
        )

    else:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
             np.clip(boxes[:, 1], 0, window[0])[:, None],
             np.clip(boxes[:, 2], 0, window[1])[:, None],
             np.clip(boxes[:, 3], 0, window[1])[:, None],
             np.clip(boxes[:, 4], 0, window[2])[:, None],
             np.clip(boxes[:, 5], 0, window[2])[:, None]), 1
        )

    return boxes