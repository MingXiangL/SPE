# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision
import pdb
from .coco import build as build_coco
from .voc import build as build_voc
from .voc_voc import build as build_voc_voc

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, dataset_file, args):
    if dataset_file == 'coco':
        args.num_classes = 90
        return build_coco(image_set, args)
        
    if dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    
    print(f'dataset file: {dataset_file}')
    
    if 'voc' in dataset_file:
        args.num_classes = 20

        if dataset_file == 'voc':
            return build_voc(image_set, args)

        if dataset_file == 'voc12':
            if image_set == 'val':
                return build_voc(image_set, args)
            else:
                dataset1 = build_voc(image_set, args, anno="pascal_train2012.json")
                dataset2 = build_voc(image_set, args, anno="pascal_val2012.json")
                return torch.utils.data.ConcatDataset([dataset1, dataset2])

        if dataset_file =='voc_voc':
            return build_voc_voc(image_set, args)
        
    raise ValueError(f'dataset {dataset_file} not supported')
