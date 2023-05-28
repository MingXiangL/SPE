# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
from json.encoder import py_encode_basestring
import math
import os
import sys
from typing import Iterable
import time
import torch

import pdb
import util.misc as utils

from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from cams_deit import resize_cam, get_bboxes, blend_cam, tensor2image, draw_gt_bbox, AveragePrecisionMeter, bgrtensor2image, draw_gt_bbox, get_multi_bboxes
import copy
import torchvision
from util.misc import all_gather, get_rank

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        pseudo_label = get_pseudo_label(outputs, samples, targets, args)
        for t, p in zip(targets, pseudo_label):
            t.update(p)
        loss_dict = criterion(outputs, targets)
        weight_dict = copy.deepcopy(criterion.weight_dict)

        if epoch < 4:
            for k in weight_dict:
                if not 'img_label' in k:
                    weight_dict[k] = 0.0

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        for key in loss_dict:
            if key in weight_dict:
                print(f'loss: {loss_dict[key]}, weight: {weight_dict[key]}')
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_refine(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, postprocessors=None, criterion_refine=None):
    
    model.train()
    criterion.train()
    criterion_refine.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    weight_dict = copy.deepcopy(criterion.weight_dict)
    rf_header = 'ref'
    weight_dict = get_refine_weight_dict(weight_dict, args.num_refines, header=rf_header)
    # if epoch % 20 == 0:
    #     args.bbox_loss_coef /= 10
    #     args.giou_loss_coef /= 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        pseudo_label = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args)

        for t, p in zip(targets, pseudo_label):
            t.update(p)

        pseudo_label_refine = get_refinements_pseudo_label(outputs, 
                                                    samples, targets, args, postprocessors)
        # print(f'outputs[0].keys(): {outputs[0].keys()}')
        # pdb.set_trace()
        loss_dict = criterion(outputs[0], targets)
        for rf, out in outputs.items():
            if rf == 0:
                continue
            loss_dict_rf = criterion_refine(out, pseudo_label_refine[rf])
            for k, v in loss_dict_rf.items():
                key = f'{rf_header}_{rf}_{k}'
                loss_dict[key] = v
        if epoch < 7:
            for k in weight_dict:
                if not ('img_label' in k or 'drloc' in k):
                    weight_dict[k] = 0.0

        if epoch < 15:
            for k in weight_dict:
                if rf_header in k:
                    weight_dict[k] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_refine_match(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, postprocessors=None, criterion_refine=None):
    
    model.train()
    criterion.train()
    criterion_refine.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    weight_dict = copy.deepcopy(criterion.weight_dict)
    rf_header = 'ref'
    weight_dict = get_refine_weight_dict(weight_dict, args.num_refines, header=rf_header)
    # if epoch % 20 == 0:
    #     args.bbox_loss_coef /= 10
    #     args.giou_loss_coef /= 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        pseudo_label = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args)

        for t, p in zip(targets, pseudo_label):
            t.update(p)
        
        pdb.set_trace()
        pseudo_label_refine = get_refinements_pseudo_label(outputs, 
                                                    samples, targets, args, postprocessors)
        loss_dict, match_indices = criterion(outputs[0], targets)
        for rf, out in outputs.items():
            if rf == 0:
                continue
            loss_dict_rf, match_indices = criterion_refine(out, pseudo_label_refine[rf])
            for k, v in loss_dict_rf.items():
                key = f'{rf_header}_{rf}_{k}'
                loss_dict[key] = v
        if epoch < 4:
            for k in weight_dict:
                if not 'img_label' in k:
                    weight_dict[k] = 0.0

        if epoch < 15:
            for k in weight_dict:
                if header in k:
                    weight_dict[k] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_refine_weight_dict(weight_dict, num_refinements, header='rf'):
    tmp_weight_dict = copy.deepcopy(weight_dict)
    org_keys = tmp_weight_dict.keys()
    for rf in range(1, num_refinements+1):
        for key in org_keys:
            new_key = f'{header}_{rf}_{key}'
            weight_dict[new_key] = weight_dict[key]

    return weight_dict


@torch.no_grad()
def get_refinements_pseudo_label(outputs, samples, targets, args, postprocessors):
    targets_refine = {}
    # targets_refine[1] = output_to_pseudo_label(outputs[0], samples, targets, args, postprocessors)
    for k, v in outputs.items():
        if k == args.num_refines:
            break
        targets_refine[k+1] = output_to_pseudo_label(v, samples, targets, args, postprocessors)
    
    return targets_refine

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def normalize_bbox(boxes, image_size):
    h, w = image_size
    boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
    return boxes


@torch.no_grad()
def output_to_pseudo_label(outputs, samples, targets, args, postprocessors):
    # device = samples.tensors.get_device()
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    pred_results = postprocessors['bbox'](outputs, orig_target_sizes, targets)
    Pseudo_labels= []
    for idx, result in enumerate(pred_results):
        Pseudo_labels.append(copy.deepcopy(targets[idx]))
        det_cls = result['labels'].detach().clone()
        det_box = result['boxes'].detach().clone()
        det_score=result['scores'].detach().clone()
        Pseudo_labels[-1].update({f'labels':det_cls, 
                        f'boxes': det_box, 
                        f'scores': det_score})
    return Pseudo_labels


@torch.no_grad()
def get_pseudo_label(outputs, samples, targets, args):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes

    device = samples.tensors.get_device()
    cams = outputs['cams_cls']
    cls_logits = outputs['x_logits']
    Pseudo_labels = []
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        image_label_i = targets[batch_i]['img_label'].data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].sigmoid().data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].reshape(-1)
        estimated_bbox = []
        estimated_class= []
        for class_i in range(args.num_classes):
            if image_label_i[class_i] > 0:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = torch.tensor(bbox)
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                estimated_class.append(class_i + 1)
        estimated_bbox = torch.stack(estimated_bbox).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)
    
        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'labels': estimated_class})
    
    return Pseudo_labels


@torch.no_grad()
def get_pseudo_label_multi_boxes(outputs, samples, targets, args):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x[...,0], x[...,1], x[...,2], x[...,3]
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes

    device = samples.tensors.get_device()
    cams = outputs['cams_cls']
    cls_logits = outputs['x_logits']
    Pseudo_labels = []
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        image_label_i = targets[batch_i]['img_label'].data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].sigmoid().data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].reshape(-1)
        estimated_bbox = []
        estimated_class= []
        for class_i in range(args.num_classes):
            if image_label_i[class_i] > 0:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                # bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = get_multi_bboxes(cam_i, cam_thr=args.cam_thr, area_ratio=args.multi_box_ratio)
                bbox = torch.tensor(bbox)
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                for _ in range(bbox.shape[0]):
                    estimated_class.append(class_i + 1)
        estimated_bbox = torch.cat(estimated_bbox, dim=0).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)

        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'labels': estimated_class})
    
    return Pseudo_labels


@torch.no_grad()
def get_pseudo_label_multi_boxes_voc(outputs, samples, targets, args):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x[...,0], x[...,1], x[...,2], x[...,3]
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes

    device = samples.tensors.get_device()
    cams = outputs['cams_cls']
    cls_logits = outputs['x_logits']
    Pseudo_labels = []
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        image_label_i = targets[batch_i]['label'].tolist()
        image_score_i = cls_logits[batch_i].sigmoid().data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].reshape(-1)
        estimated_bbox = []
        estimated_class= []
        for class_i in range(args.num_classes):
            if image_label_i[class_i] > 0:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                # bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = get_multi_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = torch.tensor(bbox)
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                for _ in range(bbox.shape[0]):
                    estimated_class.append(class_i + 1)
        estimated_bbox = torch.cat(estimated_bbox, dim=0).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)

        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'labels': estimated_class})
    
    return Pseudo_labels


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

def train_one_epoch_refine_coco(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, postprocessors=None, criterion_refine=None):
    
    model.train()
    criterion.train()
    criterion_refine.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    weight_dict = copy.deepcopy(criterion.weight_dict)
    rf_header = 'ref'
    weight_dict = get_refine_weight_dict(weight_dict, args.num_refines, header=rf_header)
    # if epoch % 20 == 0:
    #     args.bbox_loss_coef /= 10
    #     args.giou_loss_coef /= 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        pseudo_label = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args)

        for t, p in zip(targets, pseudo_label):
            t.update(p)

        pseudo_label_refine = get_refinements_pseudo_label(outputs, 
                                                    samples, targets, args, postprocessors)
        loss_dict = criterion(outputs[0], targets)
        for rf, out in outputs.items():
            if rf == 0:
                continue
            loss_dict_rf = criterion_refine(out, pseudo_label_refine[rf])
            for k, v in loss_dict_rf.items():
                key = f'{rf_header}_{rf}_{k}'
                loss_dict[key] = v
        if epoch < 1:
            for k in weight_dict:
                if not 'img_label' in k:
                    weight_dict[k] = 0.0

        if epoch < 1:
            for k in weight_dict:
                if header in k:
                    weight_dict[k] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_refinements(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, refine_stage=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs_from_model = model(samples)
        outputs = outputs_from_model[refine_stage]['aux_outputs'][-1]
        loss_criterion = copy.deepcopy(criterion.losses)
        criterion.losses = ['labels', 'boxes', 'cardinality']
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        criterion.losses = loss_criterion
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        for i, r in enumerate(results):
            pred_boxes = r['boxes']
            pred_scores= r['scores']
            pred_labels= r['labels']
            pred_classes = r['labels'].unique()
            keep_boxes = []
            keep_scores= []
            keep_labels= []
            for pc in pred_classes.tolist():
                keep_idx = (r['labels'] == pc).nonzero(as_tuple=False).reshape(-1)
                cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
                keep_box_idx = torchvision.ops.nms(cls_pred_boxes, cls_pred_score, iou_threshold=0.5)
                keep_boxes.append(cls_pred_boxes[keep_box_idx])
                keep_scores.append(cls_pred_score[keep_box_idx])
                keep_labels.append(cls_pred_labels[keep_box_idx])

            results[i]['boxes'] = torch.cat(keep_boxes)
            results[i]['scores'] = torch.cat(keep_scores)
            results[i]['labels'] = torch.cat(keep_labels)
            
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_refinements_specific_layer(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, refine_stage=0, output_layer=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs_from_model = model(samples)
        outputs = outputs_from_model[refine_stage]['aux_outputs'][output_layer]
        criterion.losses = losses = ['labels', 'boxes', 'cardinality']
        # nms_pred_boxes = torch.stack([torchvision.ops.nms(b) for b in pred_boxes], device=pred_boxes.device)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        for i, r in enumerate(results):
            pred_boxes = r['boxes']
            pred_scores= r['scores']
            pred_labels= r['labels']
            pred_classes = r['labels'].unique()
            keep_boxes = []
            keep_scores= []
            keep_labels= []
            for pc in pred_classes.tolist():
                keep_idx = (r['labels'] == pc).nonzero().reshape(-1)
                cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
                keep_box_idx = torchvision.ops.nms(cls_pred_boxes, cls_pred_score, iou_threshold=0.5)
                keep_boxes.append(cls_pred_boxes[keep_box_idx])
                keep_scores.append(cls_pred_score[keep_box_idx])
                keep_labels.append(cls_pred_labels[keep_box_idx])

            results[i]['boxes'] = torch.cat(keep_boxes)
            results[i]['scores'] = torch.cat(keep_scores)
            results[i]['labels'] = torch.cat(keep_labels)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

