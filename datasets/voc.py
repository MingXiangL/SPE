# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import torch.nn.functional as F
import pdb

import datasets.transforms as T
import torchvision.transforms as transforms


# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms, return_masks):
#         super(CocoDetection, self).__init__(img_folder, ann_file)
#         self._transforms = transforms
#         self.prepare = ConvertCocoPolysToMask(return_masks)

#     def __getitem__(self, idx):
#         img, target = super(CocoDetection, self).__getitem__(idx)
#         image_id = self.ids[idx]
#         target = {'image_id': image_id, 'annotations': target}
#         img, target = self.prepare(img, target)
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#         return img, target


# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         rles = coco_mask.frPyObjects(polygons, height, width)
#         mask = coco_mask.decode(rles)
#         if len(mask.shape) < 3:
#             mask = mask[..., None]
#         mask = torch.as_tensor(mask, dtype=torch.uint8)
#         mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks


# class ConvertCocoPolysToMask(object):
#     def __init__(self, return_masks=False):
#         self.return_masks = return_masks

#     def __call__(self, image, target):
#         w, h = image.size

#         image_id = target["image_id"]
#         image_id = torch.tensor([image_id])

#         anno = target["annotations"]

#         anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

#         boxes = [obj["bbox"] for obj in anno]
#         # guard against no boxes via resizing
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         boxes[:, 2:] += boxes[:, :2]
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)

#         classes = [obj["category_id"] for obj in anno]
#         classes = torch.tensor(classes, dtype=torch.int64)

#         if self.return_masks:
#             segmentations = [obj["segmentation"] for obj in anno]
#             masks = convert_coco_poly_to_mask(segmentations, h, w)

#         keypoints = None
#         if anno and "keypoints" in anno[0]:
#             keypoints = [obj["keypoints"] for obj in anno]
#             keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
#             num_keypoints = keypoints.shape[0]
#             if num_keypoints:
#                 keypoints = keypoints.view(num_keypoints, -1, 3)

#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         boxes = boxes[keep]
#         classes = classes[keep]
#         if self.return_masks:
#             masks = masks[keep]
#         if keypoints is not None:
#             keypoints = keypoints[keep]

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = classes
#         if self.return_masks:
#             target["masks"] = masks
#         target["image_id"] = image_id
#         if keypoints is not None:
#             target["keypoints"] = keypoints

#         # for conversion to coco api
#         area = torch.tensor([obj["area"] for obj in anno])
#         iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
#         target["area"] = area[keep]
#         target["iscrowd"] = iscrowd[keep]

#         target["orig_size"] = torch.as_tensor([int(h), int(w)])
#         target["size"] = torch.as_tensor([int(h), int(w)])

#         return image, target
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset='voc'):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.dataset = dataset
        self.prepare = ConvertCocoPolysToMask(return_masks, dataset=self.dataset)
        
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, dataset="voc"):
        self.return_masks = return_masks
        self.dataset = dataset
        if self.dataset == "coco":
            self.num_classes = 90
        else:
            self.num_classes = 20

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        classes_one_hot = F.one_hot(classes-1, self.num_classes).sum(dim=0).clamp(0,1).long()
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        target["img_label"] = classes_one_hot

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_specific_size(image_set, max_size=1333):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales_old = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    Random_size = [400, 500, 600]
    Crop_size = [384, 600]
    Random_size = [(r * max_size // 1333) for r in Random_size] 
    Crop_size = [(c * max_size // 1333) for c in Crop_size] 
    scales = []
    
    for s in scales_old:
        scales.append(s * max_size // 1333)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(Random_size),
                    T.RandomSizeCrop(*Crop_size),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.Resize((max_size,max_size), max_size=max_size),
            T.RandomResize([800*max_size//1333], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_specific_size_fixed(image_set, max_size=1333):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales_old = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    Random_size = [384, 448, 512]
    Crop_size = [384, 600]
    # Random_size = [(r * max_size // 1333) for r in Random_size] 
    Crop_size = [(c * max_size // 1333) for c in Crop_size] 
    scales = []
    
    for s in scales_old:
        scales.append(s * max_size // 1333)
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomResizeSpecific(sizes=Random_size),
            T.Resize((max_size,max_size), max_size=max_size),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=max_size),
            #     T.Compose([
            #         T.RandomResize(Random_size),
            #         T.RandomSizeCrop(*Crop_size),
            #         T.RandomResize(scales, max_size=max_size),
            #     ])
            # ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize((max_size, max_size), max_size=max_size),
            # T.RandomResize([800*max_size//1333], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

# def build(image_set, args):
#     root = Path(args.coco_path)
#     assert root.exists(), f'provided COCO path {root} does not exist'
#     mode = 'instances'
#     PATHS = {
#         "train": (root / "images" / "train2017", root / "annotations" / f'{mode}_train2017.json'),
#         "val": (root / "images" / "val2017", root / "annotations" / f'{mode}_val2017.json'),
#         "test": (root / "images" / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
#     }

#     img_folder, ann_file = PATHS[image_set]
#     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
#     return dataset


def build(image_set, args, anno="voc_07_12_trainval.json"):
    root = Path(args.coco_path)
    assert root.exists(), f'provided VOC path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images", root / "annotations" / anno),
        "val": (root / "images", root / "annotations" / "voc_2007_test.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    # if args.eval_size == 600:
    #     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms600(image_set, aug_policy=args.aug_policy), return_masks=False, dataset='voc')
    # elif args.eval_size == 384:
    #     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms384(image_set, aug_policy=args.aug_policy), return_masks=False, dataset='voc')
    # else:
    if args.fixed_size:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_specific_size_fixed(image_set, max_size=args.max_size), return_masks=False)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_specific_size(image_set, max_size=args.max_size), return_masks=False)

    return dataset

def build_voc_psuedo(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided VOC path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images", "/home/LiaoMingxiang/Workspace/weak_det/TransLocVOC3/data/voc_0712_psuedo_coco/voc_0712_trainval.json"),
        "val": (root / "images", root / "annotations" / "voc_2007_test.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    # if args.eval_size == 600:
    #     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms600(image_set, aug_policy=args.aug_policy), return_masks=False, dataset='voc')
    # elif args.eval_size == 384:
    #     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms384(image_set, aug_policy=args.aug_policy), return_masks=False, dataset='voc')
    # else:
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_specific_size(image_set, max_size=400), return_masks=False)
 
    return dataset