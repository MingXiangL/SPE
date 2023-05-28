# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
from torch._C import DeviceObjType, device
import torch.nn.functional as F
from torch import nn

import copy
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import pdb
from .cait_backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class ConditionalDETR_Refine(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, num_refines=1):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_refines = num_refines
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_refines + 1)])
        self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_refines + 1)])
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.queries_embed_refine = nn.ModuleList([nn.Embedding(num_queries, hidden_dim) for _ in range(num_refines)])

        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed in self.class_embed:
            class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features['x_patch'].decompose()
        assert mask is not None
        Hs, references = self.transformer(src, mask, self.query_embed.weight, pos[-1],
                                          queries_embed_refine=self.queries_embed_refine)

        references_before_sigmoid = [inverse_sigmoid(r) for r in references]
        # reference_before_sigmoid = inverse_sigmoid(reference)
        out = {}
        for refine_idx in range(self.num_refines + 1):
            bbox_embed = self.bbox_embed[refine_idx]
            class_embed = self.class_embed[refine_idx]
            hs = Hs[refine_idx]
            reference_before_sigmoid = references_before_sigmoid[refine_idx]
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = bbox_embed(hs[lvl])
                tmp[..., :2] += reference_before_sigmoid
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

            outputs_class = class_embed(hs)
            out_refine = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], **features}
            if self.aux_loss:
                out_refine['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            out[refine_idx] = out_refine
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, gamma, box_jitter):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.gamma = gamma
        self.eos_coef = 0.1
        self.hung_match_ratio = getattr(matcher, 'match_ratio', 1)
        self.box_jitter = box_jitter
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def update_hung_match_ratio(self, ratio=5):
        assert hasattr(self.matcher, 'match_ratio')
        self.matcher.match_ratio = ratio
        self.hung_match_ratio = ratio

    def loss_img_label(self, outputs, targets, *args, **kwargs):
        """Multi-Label Image Classification loss"""
        assert 'x_logits' in outputs and 'x_cls_logits' in outputs
        logits = outputs['x_logits']
        tokens_logits = outputs['x_cls_logits']
        target_class = torch.stack([t["img_label"] for t in targets]).to(logits.get_device()).float()
        loss_label = F.binary_cross_entropy_with_logits(logits, target_class)
        loss_label_tokens = F.binary_cross_entropy_with_logits(tokens_logits, target_class)
        losses = {"img_label_logits": loss_label, "img_label_logits_tokens": loss_label_tokens}

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        target_classes_weight = torch.ones_like(target_classes_onehot, device=src_logits.device)

        # Code for filtering with IoU
        # pred_boxes = outputs['pred_boxes']
        # EPS = 1e-6
        # for i, t in enumerate(targets):
        #     target_boxes = t['boxes']

        #     ious, _ = box_ops.box_iou(pred_boxes[i], target_boxes)
        #     neg_mask = ((ious >= EPS).sum(1) == 0)
        #     target_classes_weight[i, neg_mask] = 0

        loss_ce = self.weighted_sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                   weights=target_classes_weight, alpha=self.focal_alpha,
                                                   gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_ce(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels_ce': self.loss_labels_ce,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'image_label': self.loss_img_label
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_jittered_box(self, box, box_jitter_x, box_jitter_y, box_jitter, cnt_jitter):
        offset_cx = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(box_jitter_x[0], box_jitter_x[1]) * box[0, 2]
        offset_cy = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(box_jitter_y[0], box_jitter_y[1]) * box[0, 3]
        offset_w = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(-box_jitter, box_jitter) * box[0, 2]
        offset_h = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(-box_jitter, box_jitter) * box[0, 3]
        offset = torch.cat([offset_cx, offset_cy, offset_w, offset_h], dim=1)
        offset_box = box + offset
        iou, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(offset_box), box_ops.box_cxcywh_to_xyxy(box))
        keep_idx = torch.where(iou.reshape(-1) > 0.7)[0]
        min_keep_cnt = cnt_jitter if cnt_jitter < keep_idx.numel() else keep_idx.numel()
        box_repeat = box.repeat(cnt_jitter, 1)
        box_repeat[:min_keep_cnt] = offset_box[keep_idx[:min_keep_cnt]]
        return box_repeat


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        targets_cp = copy.deepcopy(targets)
        if self.training:
            for i in range(len(targets_cp)):
                # targets_cp[i]['boxes'].repeat(self.hung_match_ratio, 1)
                # targets_cp[i]['labels'].repeat(self.hung_match_ratio)
                # targets_cp[i]['boxes'] = targets_cp[i]['boxes'].repeat(self.hung_match_ratio, 1)
                # targets_cp[i]['labels'] = targets_cp[i]['labels'].repeat(self.hung_match_ratio)
                # if 'scores' in targets_cp[i]:
                #     targets_cp[i]['scores'] = targets[i]['scores'].repeat(self.hung_match_ratio)
                boxes_repeat = []
                for j in range(len(targets_cp[i]['labels'])):
                    box_j = targets_cp[i]['boxes'][j].reshape(1,4)
                    # scale_cx = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    # scale_cy = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    # scale_w = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    # scale_h = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    # scale = torch.cat([scale_cx, scale_cy, scale_w, scale_h], dim=1)
                    # scale_box_j = scale * box_j
#                     offset_cx = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(-self.box_jitter, self.box_jitter) * box_j[0,2]
#                     offset_cy = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(-self.box_jitter, self.box_jitter) * box_j[0,3]
#                     offset_w = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(-self.box_jitter, self.box_jitter) * box_j[0,2]
#                     offset_h = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(-self.box_jitter, self.box_jitter) * box_j[0,3]
#                     offset = torch.cat([offset_cx, offset_cy, offset_w, offset_h], dim=1)
#                     offset_box_j = box_j + offset
#                     iou_j, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(offset_box_j), box_ops.box_cxcywh_to_xyxy(box_j))
#                     keep_idx = torch.where(iou_j.reshape(-1) > 0.7)[0]
#                     min_keep_cnt = self.hung_match_ratio-1 if (self.hung_match_ratio-1) < keep_idx.numel() else keep_idx.numel()
#                     box_j_repeat = box_j.repeat(self.hung_match_ratio,1)
#                     box_j_repeat[:min_keep_cnt] = offset_box_j[keep_idx[:min_keep_cnt]]
                    cnt_jitter_quarter = int((self.hung_match_ratio-1)/4)
                    box_j_repeat1 = self.get_jittered_box(box_j, (-self.box_jitter, 0.0), (-self.box_jitter, 0.0), self.box_jitter, cnt_jitter_quarter)
                    box_j_repeat2 = self.get_jittered_box(box_j, (0.0, self.box_jitter), (-self.box_jitter, 0.0), self.box_jitter, cnt_jitter_quarter)
                    box_j_repeat3 = self.get_jittered_box(box_j, (-self.box_jitter, 0.0), (0.0, self.box_jitter), self.box_jitter, cnt_jitter_quarter)
                    box_j_repeat4 = self.get_jittered_box(box_j, (0.0, self.box_jitter), (0.0, self.box_jitter), self.box_jitter, cnt_jitter_quarter)
                    box_j_repeat = torch.cat([box_j,box_j_repeat1,box_j_repeat2,box_j_repeat3,box_j_repeat4],dim=0)
                    boxes_repeat.append(box_j_repeat)
                targets_cp[i]['boxes'] = torch.cat(boxes_repeat)
                targets_cp[i]['labels'] = targets_cp[i]['labels'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio).reshape(-1,)
                if 'scores' in targets_cp[i]:
                    targets_cp[i]['scores'] = targets_cp[i]['scores'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio).reshape(-1,)

        indices = self.matcher(outputs_without_aux, targets_cp)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets_cp)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets_cp, indices, num_boxes))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets_cp)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if loss == 'image_label':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets_cp, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def weighted_sigmoid_focal_loss(self, inputs, targets, num_boxes, weights, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        EPS = 1e-5
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = p_t.clamp(EPS, 1 - EPS)
        loss = weights * ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes


class SetCriterionRefine(SetCriterion):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # target_classes_onehot.shape: B x num_queries x num_classes
        # indices: []
        avg_score = torch.tensor([t['scores'].mean() for t in targets], device=src_logits.device).reshape(-1, 1, 1)
        target_classes_weight = torch.ones_like(target_classes_onehot, device=src_logits.device) * avg_score
        for batch_i in range(len(indices)):
            R, C = indices[batch_i][0], indices[batch_i][1]
            target_classes_weight[batch_i, R, :] = (
                        targets[batch_i]['scores'][C].unsqueeze(-1).repeat(1, src_logits.shape[-1]) * 3).clamp(max=1.0)

        loss_ce = self.weighted_sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                   weights=target_classes_weight,
                                                   alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        boxes_weight = torch.cat([t['scores'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox * boxes_weight.reshape(-1, 1)
        losses = {}

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def weighted_sigmoid_focal_loss(self, inputs, targets, num_boxes, weights, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        EPS = 1e-5
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = p_t.clamp(EPS, 1 - EPS)
        loss = weights * ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox).clamp(min=0)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PostProcessRefine(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        device = out_logits.get_device()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # TODO: 1. 数据集label是从0开始还是从1开始， 从1开始
        #       2. 这个post processor改为为每个类找到得分最高的box
        prob = out_logits.sigmoid()

        top_values, top_indexes = torch.max(prob, dim=1)
        top_boxes = torch.gather(out_bbox, 1, top_indexes.unsqueeze(-1).repeat(1, 1, 4))
        top_labels = torch.arange(out_logits.shape[2]).unsqueeze(0).repeat(out_logits.shape[0], 1)
        scores, boxes, labels = [], [], []
        for ii in range(len(targets)):
            tmp_labels, tmp_scores, tmp_boxes = [], [], []
            for cc in range(out_logits.shape[2]):
                if cc in targets[ii]['labels']:
                    tmp_labels.append(cc)
                    tmp_scores.append(top_values[ii][cc].reshape(-1))
                    tmp_boxes.append(top_boxes[ii][cc].reshape(1, -1))
            labels.append(torch.tensor(tmp_labels).to(device))
            scores.append(torch.cat(tmp_scores, dim=0))
            boxes.append(torch.cat(tmp_boxes, dim=0))

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class PostProcessRefineMulti(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        device = out_logits.get_device()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # TODO: 1. 数据集label是从0开始还是从1开始， 从1开始
        #       2. 这个post processor改为为每个类找到得分最高的box
        prob = out_logits.sigmoid()

        top_values, top_indexes = torch.max(prob, dim=1)
        keep_idx = (prob >= 0.5 * top_values.unsqueeze(1).expand_as(prob))
        scores, boxes, labels = [], [], []
        for ii in range(len(targets)):
            tmp_labels, tmp_scores, tmp_boxes = [], [], []
            for cc in range(out_logits.shape[2]):
                if cc in targets[ii]['labels']:
                    keep_idx_c = keep_idx[ii, :, cc].nonzero(as_tuple=False).reshape(-1)
                    tmp_scores.append(prob[ii, keep_idx_c, cc])
                    tmp_boxes.append(out_bbox[ii, keep_idx_c])
                    tmp_labels += [cc for _ in range(keep_idx_c.shape[0])]

            labels.append(torch.tensor(tmp_labels).to(device))
            scores.append(torch.cat(tmp_scores, dim=0))
            boxes.append(torch.cat(tmp_boxes, dim=0))
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 21 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = ConditionalDETR_Refine(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        num_refines=args.num_refines,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    matcher_refine = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'img_label_logits': args.img_label_loss_coef,
                   'img_label_logits_tokens': args.img_label_tokens_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'image_label']
    losses_refine = ['labels', 'boxes', 'cardinality']

    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, gamma=args.focal_gamma, box_jitter=args.box_jitter)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    # refine_postprocessors = {'bbox': PostProcessRefineMulti()}
    refine_postprocessors = {'bbox': PostProcessRefine()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    criterion_refine = SetCriterionRefine(num_classes, matcher=matcher_refine, weight_dict=weight_dict,
                                          focal_alpha=args.focal_alpha, losses=losses_refine, gamma=args.focal_gamma, box_jitter=args.box_jitter)
    criterion_refine.to(device)

    # return model, criterion, postprocessors
    return model, criterion, criterion_refine, postprocessors, refine_postprocessors
