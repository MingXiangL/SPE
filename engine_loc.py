# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math

from torch.nn.modules.loss import BCELoss
import os
import cv2
import torch

from util import box_ops
import util.misc as utils
from cams_deit import resize_cam, get_bboxes, blend_cam, tensor2image, draw_gt_bbox, AveragePrecisionMeter, bgrtensor2image, draw_gt_bbox, get_multi_bboxes
import numpy as np
import pdb
import time
import timm
import torchvision
import copy

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

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
def evaluate_det_voc(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, cache_dir, with_flip=False, *kargs, **kwargs):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{gettime()} => Test:'
    
    loc_all_boxes = [[[] for _ in range(len(data_loader.dataset))]
               for _ in range(len(data_loader.dataset.classes))]
    loc_all_boxes_loc = [[[] for _ in range(len(data_loader.dataset))]
            for _ in range(len(data_loader.dataset.classes))]

    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        batch_size = samples.tensors.shape[0]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if with_flip:
            samples_coupled = copy.deepcopy(samples)
            samples_coupled = torch.cat((samples.tensors, torchvision.transforms.functional.hflip(samples.tensors)), dim=0)
            outputs_from_model_coupled = model(samples_coupled)
            outputs_coupled = outputs_from_model_coupled[0]
            outputs = decouple_output(outputs_coupled, bs=batch_size)
        else:
            outputs = model(samples)
            outputs = outputs[0]
                
        orig_target_sizes = torch.stack([t["image_size"].flip(0) for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, 300)
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

    corloc = data_loader.dataset.evaluate_discovery(loc_all_boxes_loc, cache_dir)
    data_loader.dataset.evaluate_detections(loc_all_boxes, cache_dir)
    # torch.save(attn_list, save_path_attn)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return 


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
        
