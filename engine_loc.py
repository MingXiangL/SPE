# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math

from torch.nn.modules.loss import BCELoss
from models.layers import weight_init
import os
import sys
from typing import Iterable
import cv2
import torch

from util import box_ops
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
import torch.nn.functional as F
from cams_deit import resize_cam, get_bboxes, blend_cam, tensor2image, draw_gt_bbox, AveragePrecisionMeter, bgrtensor2image, draw_gt_bbox, get_multi_bboxes, get_bboxes_ivr
from engine import get_pseudo_label_multi_boxes_voc
import numpy as np
import pdb
import time
import timm
import torchvision
import copy
# from torch.cuda.amp import GradScaler, autocast
# from apex import amp
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

# scaler = GradScaler()
# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
#     model.train()
#     criterion.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100

#     # AMP
#     scaler = torch.cuda.amp.GradScaler()

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         with torch.cuda.amp.autocast():

#             outputs = model(samples)            
#             loss_dict = criterion(outputs, targets)
#             weight_dict = criterion.weight_dict
#             losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#             # reduce losses over all GPUs for logging purposes
#             loss_dict_reduced = utils.reduce_dict(loss_dict)
#             loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                         for k, v in loss_dict_reduced.items()}
#             loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                         for k, v in loss_dict_reduced.items() if k in weight_dict}
#             losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

#             loss_value = losses_reduced_scaled.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         scaler.scale(losses).backward()

#         if max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

#         scaler.step(optimizer)
#         scaler.update()

#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    BCELoss = torch.nn.BCEWithLogitsLoss()
    # count = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        # count += 1
        # if count == 10: break

        samples = samples.to(device)
        targets0 = targets
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        # if "logits" in outputs:
        #     labels = torch.stack([t['img_label'] for t in targets]).float()
        #     loss_dict.update({'loss_cls': BCELoss(outputs['logits'], labels)})
        #     # loss_dict.update({'loss_cls': F.binary_cross_entropy_with_logits(outputs['logits'], labels)})
        #     criterion.weight_dict['loss_cls'] = 1.0
        # criterion.weight_dict['loss_ce'] = 0
        # criterion.weight_dict['loss_giou'] = 0
        # criterion.weight_dict['loss_bbox'] = 0
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # losses = loss_dict['loss_cls']
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

def train_one_epoch_voc(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    BCELoss = torch.nn.BCEWithLogitsLoss()
    # count = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        # count += 1
        # if count == 10: break
        samples = samples.to(device)
        targets0 = targets
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = {}
        
        # loss_dict['img_label'] = loss_img_label
        loss_dict = criterion(outputs, targets)
        # if "logits" in outputs:
        #     labels = torch.stack([t['img_label'] for t in targets]).float()
        #     loss_dict.update({'loss_cls': BCELoss(outputs['logits'], labels)})
        #     # loss_dict.update({'loss_cls': F.binary_cross_entropy_with_logits(outputs['logits'], labels)})
        #     criterion.weight_dict['loss_cls'] = 1.0
        # criterion.weight_dict['loss_ce'] = 0
        # criterion.weight_dict['loss_giou'] = 0
        # criterion.weight_dict['loss_bbox'] = 0
        # weight_dict = criterion.weight_dict
        # weight_dict = {'empty_loss': 0, 'img_label': 1}
        weight_dict = {'img_label': 1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # losses = loss_dict['loss_cls']
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
        
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_amp(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # count = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        # count += 1
        # if count == 10: break

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
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

    with amp.scale_loss(losses, optimizer) as scaled_loss:
        scaled_loss.backward()

    # losses.backward()
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
    optimizer.step()

    metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
    metric_logger.update(class_error=loss_dict_reduced['class_error'])
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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


    for samples, targets in metric_logger.log_every(data_loader, 256, header):
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


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
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

def gettime():
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def get_loc_box(image, cams, cls_logits, targets, loc_all_boxes, img_id, num_classses=20, cam_thr=0.2, save_image=False, dataset=None):
    T = gettime()
    image_from_tensor = bgrtensor2image(image.clone().detach().cpu(), timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
    for batch_i in range(cams.shape[0]):
        img_score =cls_logits[batch_i].sigmoid().data.cpu().numpy()
        img_label = targets[batch_i]['label']
        img_size = targets[batch_i]['image_size'].tolist()
        # img_label = targets[batch_i]['img_label']
        # img_size = targets[batch_i]['size'].tolist()
        image_i = image_from_tensor[batch_i]
        image_i = cv2.resize(image_i, (img_size[0], img_size[1]))
        img_id = int(targets[batch_i]['idx'])
        for class_i in range(num_classses):
            if img_label[class_i] == 1:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=(int(img_size[0]), int(img_size[1])))
                bbox = get_bboxes(cam_i, cam_thr=cam_thr)
                bbox_score = np.hstack((np.array(bbox).reshape(1, -1), img_score[class_i].reshape(1, -1)))
                loc_all_boxes[class_i][img_id] = bbox_score.copy()
                if save_image:
                    save_dir = os.path.join('./output', 'visulization')
                    blend, _ = blend_cam(image_i, cam_i)
                    # boxed_image = draw_gt_bbox(blend, targets[batch_i]['gt_box'], [bbox], img_score[class_i].reshape(1, -1))
                    boxed_image = draw_gt_bbox(blend, None, [bbox], img_score[class_i].reshape(1, -1))
                    save_dir = os.path.join(save_dir, 'boxed_image','visualization')
                    save_path = os.path.join(save_dir,f'{dataset._image_index[img_id]}_{CLASSES[class_i]}.jpg')
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(save_path, boxed_image)
                    
        return loc_all_boxes, img_id


def get_loc_multi_boxes(image, cams, cls_logits, targets, loc_all_boxes, img_id, num_classses=20, cam_thr=0.2, save_image=False, dataset=None):
    T = gettime()
    image_from_tensor = bgrtensor2image(image.clone().detach().cpu(), timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
    for batch_i in range(cams.shape[0]):
        img_score =cls_logits[batch_i].sigmoid().data.cpu().numpy()
        img_label = targets[batch_i]['label']
        img_size = targets[batch_i]['image_size'].tolist()
        # img_label = targets[batch_i]['img_label']
        # img_size = targets[batch_i]['size'].tolist()
        image_i = image_from_tensor[batch_i]
        image_i = cv2.resize(image_i, (img_size[0], img_size[1]))
        img_id = int(targets[batch_i]['idx'])
        for class_i in range(num_classses):
            if img_label[class_i] == 1:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=(int(img_size[0]), int(img_size[1])))
                bbox = get_multi_bboxes(cam_i, cam_thr=cam_thr)
                bbox_score = np.hstack((np.array(bboxes).reshape(-1, 4), np.tile(image_score_i[class_i].reshape(1, -1), (len(bboxes),1))))
                loc_all_boxes[class_i][img_id] = bbox_score.copy()
                if save_image:
                    save_dir = os.path.join('./output', 'visulization')
                    blend, _ = blend_cam(image_i, cam_i)
                    boxed_image = draw_gt_bbox(blend, targets[batch_i]['gt_box'], [bbox], img_score[class_i].reshape(1, -1))
                    boxed_image = draw_gt_bbox(boxed_image, None, [bbox], img_score[class_i].reshape(1, -1))
                    save_dir = os.path.join(save_dir, 'boxed_image','visualization')
                    save_path = os.path.join(save_dir,f'{dataset._image_index[img_id]}_{CLASSES[class_i]}.jpg')
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(save_path, boxed_image)
                    
        return loc_all_boxes, img_id


@torch.no_grad()
def evaluate_loc_voc(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, cache_dir, dataset):
    ap_meter_cls = AveragePrecisionMeter(difficult_examples=False)
    ap_meter_cls.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = None
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    img_id = -1
    loc_all_boxes = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(len(data_loader.dataset.classes))]
    loc_all_boxes_det = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(len(data_loader.dataset.classes))]

    BCELoss = torch.nn.BCEWithLogitsLoss()
    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)[0]
        labels = torch.stack([t['label'] for t in targets]).float()
        ap_meter_cls.add(outputs['x_logits'], labels)
        
        loss_dict = {}
        cams = outputs['cams_cls']
        cls_logits = outputs['x_logits']
        image_from_tensor = bgrtensor2image(samples.tensors.clone().detach().cpu(), timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)

        for batch_i in range(cams.shape[0]):
            img_score = cls_logits[batch_i].sigmoid().data.cpu().numpy()
            img_label = targets[batch_i]['label']
            img_size = targets[batch_i]['image_size'].tolist()
            # img_label = targets[batch_i]['img_label']
            # img_size = targets[batch_i]['size'].tolist()
            image_i = image_from_tensor[batch_i]
            image_i = cv2.resize(image_i, (img_size[0], img_size[1]))
            img_id = int(targets[batch_i]['idx'])
            for class_i in range(args.num_classes):
                if img_label[class_i] == 1:
                    cam_i = cams[batch_i, [class_i], :, :]
                    cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                    cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                    cam_i = resize_cam(cam_i, size=(int(img_size[0]), int(img_size[1])))
                    bbox = get_bboxes_ivr(cam_i, cam_thr=args.cam_thr)
                    # bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                    bbox_multi = get_multi_bboxes(cam_i, cam_thr=args.cam_thr, area_ratio=args.area_ratio)
                    bbox_score = np.hstack((np.array(bbox).reshape(1, -1), img_score[class_i].reshape(1, -1)))
                    bbox_score_multi = np.hstack((np.array(bbox_multi).reshape(-1, 4), 
                                        img_score[class_i].reshape(1, -1).repeat(len(bbox_multi), axis=0)))
                    loc_all_boxes[class_i][img_id] = bbox_score.copy()
                    loc_all_boxes_det[class_i][img_id] = bbox_score_multi.copy()
                    # if args.visualize:
                    #     save_dir = os.path.join('./output', args.vis_name)
                    #     blend, _ = blend_cam(image_i, cam_i)
                    #     # boxed_image = draw_gt_bbox(blend, targets[batch_i]['gt_box'].tolist(), None, img_score[class_i].reshape(1, -1), color1=(0,255,0))
                    #     boxed_image = draw_gt_bbox(blend, targets[batch_i]['gt_box'], [bbox], img_score[class_i].reshape(1, -1))
                    #     save_dir = os.path.join(save_dir, 'boxed_image','visualization')
                    #     save_path = os.path.join(save_dir,f'{dataset._image_index[img_id]}_{CLASSES[class_i]}.jpg')
                    #     os.makedirs(save_dir, exist_ok=True)
                        # cv2.imwrite(save_path, boxed_image)
                    
    
    ap = ap_meter_cls.value()
    print('The classification AP is')
    for index, cls in enumerate(CLASSES):
        print('Ap for {} = {:.4f}'.format(cls, 100*ap[index]))
    print('the mAP is {:.4f}'.format(100*ap.mean()))
    corloc = data_loader.dataset.evaluate_discovery(loc_all_boxes, cache_dir)
    data_loader.dataset.evaluate_detections(loc_all_boxes_det, cache_dir)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
                
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
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

    return stats, coco_evaluator, corloc



@torch.no_grad()
def evaluate_loc(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, cache_dir):
    model.eval()
    criterion.eval()
    ap_meter_cls = AveragePrecisionMeter(difficult_examples=False)
    ap_meter_cls.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    img_id = -1
    loc_all_boxes = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(len(data_loader.dataset.classes))]
    
    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        labels = torch.stack([t['img_label'] for t in targets]).float()
        ap_meter_cls.add(outputs['logits'], labels)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loc_all_boxes, img_id = get_loc_box(samples.tensors, outputs['cam'], outputs['logits'], targets, loc_all_boxes, img_id, args.num_classes, args.cam_thr, args.visualize)

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
            
        if args.visualize:
            images = tensor2image(samples.tensors)
            cam = outputs['cam'].cpu().numpy()
            for b in range(images.shape[0]):
                for cls in targets[b]['labels'].cpu().numpy():
                    cam_b = cam[b]
                    cam_b = resize_cam(cam_b[cls-1], size=(images.shape[2], images.shape[1]))
                    blend, heatmap = blend_cam(images[b], cam_b)
                    save_path = os.path.join(args.output_dir, 'visualization',f"{int(targets[b]['image_id'])}_{CLASSES[cls-1]}.jpg")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, blend)
    
    ap = ap_meter_cls.value()
    print('The classification AP is')
    for index, cls in enumerate(CLASSES):
        print('Ap for {} = {:.4f}'.format(cls, 100*ap[index]))
    print('the mAP is {:.4f}'.format(100*ap.mean()))

    corloc = data_loader.dataset.evaluate_discovery(loc_all_boxes, cache_dir)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
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

    return stats, coco_evaluator, corloc


@torch.no_grad()
def evaluate_loc_voc_det(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, cache_dir, dataset):
    model.eval()
    criterion.eval()
    ap_meter_cls = AveragePrecisionMeter(difficult_examples=False)
    ap_meter_cls.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'{gettime()} => Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    coco_evaluator = None
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    img_id = -1
    loc_all_boxes = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(len(data_loader.dataset.classes))]
    BCELoss = torch.nn.BCEWithLogitsLoss()
    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["image_size"].flip(0) for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['idx'].item(): output for target, output in zip(targets, results)}

        labels = torch.stack([t['label'] for t in targets]).float()
        ap_meter_cls.add(outputs['x_logits'], labels)
        
        loss_dict = {}
        cams = outputs['cams_cls']
        cls_logits = outputs['x_logits']
        image_from_tensor = bgrtensor2image(samples.tensors.clone().detach().cpu(), timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
        
        # for T in targets:
        #     det_cls = T['gt_box'][:,-1].int().detach().cpu().numpy()
        #     det_box = T['gt_box'][:,:-1].detach().cpu().numpy()
        #     for i, cls in enumerate(det_cls):
        #         bbox_score = np.hstack((det_box[i].reshape(1,-1), np.ones((1,1))))
        #         loc_all_boxes[cls][T['idx'].item()] = bbox_score.copy()

        for r in res:
            det_cls = res[r]['labels'].int().detach().cpu().numpy()
            det_box = res[r]['boxes'].detach().cpu().numpy()
            det_score=res[r]['scores'].detach().cpu().numpy()
            keep = torch.nonzero(res[r]['scores'] >= 0.5).reshape(-1)
            
            for idx in keep:
                bbox_score = np.hstack((det_box[idx].reshape(1, -1), det_score[idx].reshape(1, -1)))
                if isinstance(loc_all_boxes[det_cls[idx]-1][r], list):
                    loc_all_boxes[det_cls[idx]-1][r] = bbox_score.copy()
                else:
                    loc_all_boxes[det_cls[idx]-1][r] = np.vstack((loc_all_boxes[det_cls[idx]-1][r], bbox_score.copy()))
        # for batch_i in range(cams.shape[0]):
        #     img_score = cls_logits[batch_i].sigmoid().data.cpu().numpy()
        #     img_label = targets[batch_i]['label']
        #     img_size = targets[batch_i]['image_size'].tolist()
        #     # img_label = targets[batch_i]['img_label']
        #     # img_size = targets[batch_i]['size'].tolist()
        #     image_i = image_from_tensor[batch_i]
        #     image_i = cv2.resize(image_i, (img_size[0], img_size[1]))
        #     img_id = int(targets[batch_i]['idx'])
        #     for class_i in range(args.num_classes):
        #         if img_label[class_i] == 1:
        #             cam_i = cams[batch_i, [class_i], :, :]
        #             cam_i = torch.mean(cam_i, dim=0, keepdim=True)
        #             cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
        #             cam_i = resize_cam(cam_i, size=(int(img_size[0]), int(img_size[1])))
        #             bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
        #             bbox_score = np.hstack((np.array(bbox).reshape(1, -1), img_score[class_i].reshape(1, -1)))
        #             loc_all_boxes[class_i][img_id] = bbox_score.copy()
        #             if args.visualize:
        #                 save_dir = os.path.join('./output', 'visualization', args.vis_name)
        #                 os.makedirs(save_dir, exist_ok=True)
        #                 blend, _ = blend_cam(image_i, cam_i)
        #                 # boxed_image = draw_gt_bbox(blend, targets[batch_i]['gt_box'], [bbox], img_score[class_i].reshape(1, -1))
        #                 boxed_image = draw_gt_bbox(blend, targets[batch_i]['gt_box'], [bbox], img_score[class_i].reshape(1, -1))
        #                 save_dir = os.path.join(save_dir, 'boxed_image')
        #                 save_path = os.path.join(save_dir,f'{dataset._image_index[img_id]}_{CLASSES[class_i]}.jpg')
        #                 os.makedirs(save_dir, exist_ok=True)
        #                 cv2.imwrite(save_path, boxed_image)
    
    ap = ap_meter_cls.value()
    print('The classification AP is')
    for index, cls in enumerate(CLASSES):
        print('Ap for {} = {:.4f}'.format(cls, 100*ap[index]))
    print('the mAP is {:.4f}'.format(100*ap.mean()))
    corloc = data_loader.dataset.evaluate_discovery(loc_all_boxes, cache_dir)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
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

    return stats, coco_evaluator, corloc


def decouple_output(output, bs=2):
    combine_keys =  ['pred_logits', 'pred_boxes', 'x_logits', 'x_cls_logits', 'cams_cls', 'aux_outputs']
    for k in combine_keys:
        if not k in output.keys():
            continue

        v = output[k]
        idxes_pre = [i for i in range(bs)]
        idxes_pos = [i for i in range(bs,2*bs)]

        if k == 'pred_boxes' :
            v[idxes_pos,:,0] = 1 - v[idxes_pos,:,0] # x,y,w,h
        if (k=='x_logits') or (k=='x_cls_logits'):
            v1 = torch.maximum(v[idxes_pre], v[idxes_pos])
            output.update({k: v1})
            continue
        if k == 'aux_outputs':
            for i, aux in enumerate(output[k]):
                output[k][i] = decouple_output(aux, bs=bs)
            continue
        # try:
        output.update({k: torch.cat((v[idxes_pre], v[idxes_pos]), dim=1)})
        # except BaseException:
        #     print(f'output.keys(): {output.keys()}')
        #     pdb.set_trace()
    return output


@torch.no_grad()
def evaluate_det_voc(model, criterion, postprocessors, data_loader, device, output_dir, args, cache_dir, dataset):
    model.eval()
    criterion.eval()
    ap_meter_cls = AveragePrecisionMeter(difficult_examples=False)
    ap_meter_cls.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'{gettime()} => Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    coco_evaluator = None
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    img_id = -1
    loc_all_boxes = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(len(data_loader.dataset.classes))]
    loc_all_boxes_loc = [[[] for _ in range(len(data_loader.dataset))]
            for _ in range(len(data_loader.dataset.classes))]

    BCELoss = torch.nn.BCEWithLogitsLoss()
    label_min = 30
    label_max = -1
    # postprocessors['bbox'].num_keep_queries *= 2
    attn_list = []
    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        # print(f'iamge_size:{samples.tensors.shape}')
        batch_size = samples.tensors.shape[0]
        # samples_coupled = copy.deepcopy(samples)
        # samples_coupled = torch.cat((samples.tensors, torchvision.transforms.functional.hflip(samples.tensors)), dim=0)
        target_save = [{k: v.cpu() for k, v in t.items()} for t in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # outputs_from_model_coupled = model(samples_coupled)
        # outputs = outputs_from_model[refine_stage]['aux_outputs'][-1]
        # outputs_coupled = outputs_from_model_coupled[0]
        # outputs = decouple_output(outputs_coupled, bs=batch_size)

        outputs = model(samples)[0]
        orig_target_sizes = torch.stack([t["image_size"].flip(0) for t in targets], dim=0)
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

                # keep_box_idx = soft_nms_pytorch(cls_pred_boxes, cls_pred_score, device=cls_pred_boxes.device)
                keep_boxes.append(cls_pred_boxes[keep_box_idx])
                keep_scores.append(cls_pred_score[keep_box_idx])
                keep_labels.append(cls_pred_labels[keep_box_idx])

            results[i]['boxes'] = torch.cat(keep_boxes)
            results[i]['scores'] = torch.cat(keep_scores)
            results[i]['labels'] = torch.cat(keep_labels)
        
        res = {target['idx'].item(): output for target, output in zip(targets, results)}

        if 'label' in targets[0]:
            labels = torch.stack([t['label'] for t in targets]).float()
            # print(f'outputs.keys():{outputs.keys()}')
            ap_meter_cls.add(outputs['x_logits'], labels)
        
        loss_dict = {}
        cams = outputs['cams_cls']
        cls_logits = outputs['x_logits']
        image_from_tensor = bgrtensor2image(samples.tensors.clone().detach().cpu(), timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
        
        for batch_i, r in enumerate(res):
            det_cls = res[r]['labels'].int().detach().cpu().numpy()
            det_box = res[r]['boxes'].detach().cpu().numpy()
            det_score=res[r]['scores'].detach().cpu().numpy()
            image_id = int(targets[batch_i]['idx'])
            for idx in range(det_box.shape[0]):
                if det_cls[idx] == 0:
                    continue
                bbox_score = np.hstack((det_box[idx].reshape(1, -1), det_score[idx].reshape(1, -1)))
                if isinstance(loc_all_boxes[det_cls[idx]-1][image_id], list):
                    loc_all_boxes[det_cls[idx]-1][image_id] = bbox_score.copy()
                    loc_all_boxes_loc[det_cls[idx]-1][image_id] = bbox_score.copy()
                else:
                    loc_all_boxes[det_cls[idx]-1][image_id] = np.vstack((loc_all_boxes[det_cls[idx]-1][image_id], bbox_score.copy()))
                    # pdb.set_trace()
                    if loc_all_boxes_loc[det_cls[idx]-1][image_id][0, -1] < bbox_score[0, -1]:
                        loc_all_boxes_loc[det_cls[idx]-1][image_id] = bbox_score.copy()

                # raise NotImplementedError
            # pdb.set_trace()
                
    # ap = ap_meter_cls.value()
    # print('The classification AP is')
    # for index, cls in enumerate(CLASSES):
    #     print('Ap for {} = {:.4f}'.format(cls, 100*ap[index]))
    # print('the mAP is {:.4f}'.format(100*ap.mean()))
    # corloc = data_loader.dataset.evaluate_discovery(loc_all_boxes_loc, cache_dir)
    data_loader.dataset.evaluate_detections(loc_all_boxes, cache_dir)
    save_path_attn = './outputs/attn_weights.pth'
    os.makedirs(os.path.dirname(save_path_attn), exist_ok=True)
    torch.save(attn_list, save_path_attn)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
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
def evaluate_tscam_voc(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, cache_dir, dataset):
    model.eval()
    criterion.eval()
    ap_meter_cls = AveragePrecisionMeter(difficult_examples=False)
    ap_meter_cls.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'{gettime()} => Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    coco_evaluator = None
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    img_id = -1
    num_classes = 20
    loc_all_boxes = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(num_classes)]
    loc_all_boxes_loc = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(num_classes)]
    BCELoss = torch.nn.BCEWithLogitsLoss()
    label_min = 30
    label_max = -1
    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs_from_model = model(samples)
        outputs = outputs_from_model[0]
        pseudo_label = get_pseudo_label_multi_boxes_voc(outputs_from_model[0], samples, targets, args)

        orig_target_sizes = torch.stack([t["image_size"].flip(0) for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        # print(targets)
        results = pseudo_label_to_det_out(pseudo_label, orig_target_sizes)

        # pdb.set_trace()
        res = {target['idx'].item(): output for target, output in zip(targets, results)}

        labels = torch.stack([t['label'] for t in targets]).float()
        # ap_meter_cls.add(outputs['x_logits'], labels)
        
        loss_dict = {}
        cams = outputs['cams_cls']
        cls_logits = outputs['x_logits']
        image_from_tensor = bgrtensor2image(samples.tensors.clone().detach().cpu(), timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
        
        for batch_i, r in enumerate(res):
            det_cls = res[r]['labels'].int().detach().cpu().numpy()
            det_box = res[r]['boxes'].detach().cpu().numpy()
            det_score=res[r]['scores'].detach().cpu().numpy()
            image_id = int(targets[batch_i]['idx'])
            for idx in range(det_box.shape[0]):
                if det_cls[idx] == 0:
                    continue
                bbox_score = np.hstack((det_box[idx].reshape(1, -1), det_score[idx].reshape(1, -1)))
                if isinstance(loc_all_boxes[det_cls[idx]-1][image_id], list):
                    loc_all_boxes[det_cls[idx]-1][image_id] = bbox_score.copy()
                else:
                    loc_all_boxes[det_cls[idx]-1][image_id] = np.vstack((loc_all_boxes[det_cls[idx]-1][image_id], bbox_score.copy()))
                                                                            
                if isinstance(loc_all_boxes_loc[det_cls[idx]-1][image_id], list):
                    loc_all_boxes_loc[det_cls[idx]-1][image_id] = bbox_score.copy() 
                    # print('det_cls[idx]-1:', det_cls[idx]-1)
                    # print(f'loc_all_boxes_loc[det_cls[idx]-1][image_id]: {loc_all_boxes_loc[det_cls[idx]-1][image_id]}')
                else:
                    if det_score[idx] > loc_all_boxes_loc[det_cls[idx]-1][image_id][0, -1]:
                        loc_all_boxes_loc[det_cls[idx]-1][image_id] = bbox_score.copy()
                        # print(f'*********loc_all_boxes_loc[det_cls[idx]-1][image_id]: {loc_all_boxes_loc[det_cls[idx]-1][image_id]}')
    # ap = ap_meter_cls.value()
    # print('The classification AP is')
    # for index, cls in enumerate(CLASSES):
    #     print('Ap for {} = {:.4f}'.format(cls, 100*ap[index]))
    # print('the mAP is {:.4f}'.format(100*ap.mean()))
                                        
    corloc = data_loader.dataset.evaluate_discovery(loc_all_boxes_loc, cache_dir)
    data_loader.dataset.evaluate_detections(loc_all_boxes, cache_dir)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
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


def pseudo_label_to_det_out(pseudo_label, target_sizes):
    assert target_sizes.shape[1] == 2
    results = []
    for idx, p in enumerate(pseudo_label):
        img_h, img_w = target_sizes[idx]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0).unsqueeze(0)

        box = p['boxes']
        box = box_ops.box_cxcywh_to_xyxy(box).clamp(min=0)
        box = box * scale_fct
        label= p['labels']
        scores = torch.ones_like(label, device=label.device)
        res = {'scores':scores, 'labels': label, 'boxes':box}
        
        results.append(res)
    
    return results
        
