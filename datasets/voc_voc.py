from pathlib import Path

import os
import numpy as np
import subprocess
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .voc_eval import voc_eval
from .dis_eval import dis_eval
import pdb
xrange = range  # Python 3

import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import timm

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor[[2,1,0]]
        tensor = tensor * 255.
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transforms(args):
    train_transform = transforms.Compose([
        transforms.Resize(args.max_size),
        transforms.RandomCrop(args.max_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.max_size,args.max_size)),
        transforms.ToTensor(),
        transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
    ])
    test_tencrops_transform = transforms.Compose([
        transforms.Resize(args.max_size),
        transforms.TenCrop(args.max_size),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
             (transforms.ToTensor()(crop)) for crop in crops])),
    ])
    return train_transform, test_transform, test_tencrops_transform


class VOCDataset(Dataset):
    def __init__(self, root, args, is_train):
        self.root = root
        self.args = args
        self.is_train = is_train
        self.resize_size = args.max_size
        self.crop_size = args.max_size
        # self._year = [2007, 2012]
        self._year = [2007, 2012] if is_train else [2007]
        self._year_test = 2007
        self._year_list = [str(year) for year in self._year]
        # self._image_set = 'trainval'
        self._image_set = 'trainval' if is_train else 'test'
        self.name = ['voc_' + str(year) + '_' + self._image_set for year in self._year_list]
        self._devkit_path = self._get_default_path()
        self._data_path = [os.path.join(devkit_path, 'VOC' + str(year)) for devkit_path, year in zip(self._devkit_path, self._year_list)]
        self._image_dir = [os.path.join(data_path, 'JPEGImages') for data_path in self._data_path]
        self._classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._num_classes = len(self._classes)
        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': False,
                       'use_diff': False,
                       'matlab_eval': False}
        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._image_ext = '.jpg'
        self._image_index, self._image_index_data_path = self._load_image_set_index()
        self._image_label = self._load_image_label()
        self._image_gt_box = self._load_image_gt_box()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        # self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.train_transform, self.onecrop_transform, self.tencrops_transform = get_transforms(args)
        # if cfg.TEST.TEN_CROPS:
        #     self.test_transform = self.tencrops_transform
        # else:
        self.test_transform = self.onecrop_transform

        for devkit_path, data_path in zip(self._devkit_path, self._data_path):
            assert os.path.exists(devkit_path), \
                'VOCdevkit path does not exist: {}'.format(devkit_path)
            assert os.path.exists(data_path), \
                'Path does not exist: {}'.format(data_path)

    def get_image_from_idx(self, idx):
        name = self._image_index[idx]
        data_path_i = self._image_index_data_path[idx]
        image = Image.open(os.path.join(self._image_dir[data_path_i], name+self._image_ext)).convert('RGB')
        return image

    def __getitem__(self, idx):
        name = self._image_index[idx]
        data_path_i = self._image_index_data_path[idx]
        label = torch.tensor(self._image_label[idx]).float()
        gt_box = torch.tensor(self._image_gt_box[idx]).float()
        image = Image.open(os.path.join(self._image_dir[data_path_i], name+self._image_ext)).convert('RGB')

        image_size = torch.tensor(np.array(list(image.size)))

        # if self.is_train:
        #     image = self.train_transform(image)
        #     return image, {'label':label}
        # else:
        image = self.test_transform(image)

            # gt_box = " ".join(list(map(str, gt_box)))
        return image, {'label':label, 'gt_box':gt_box, 
                'image_size':image_size, 'idx': torch.tensor(idx)}
#, 'image_name':name + self._image_ext, 'data_path':data_path_i

    def __len__(self):
        return len(self._image_index)

    def _load_image_label(self):
        image_label_list = []
        for i, index in enumerate(self._image_index):
            image_label_list.append(self._load_pascal_labels(index, self._image_index_data_path[i]))
        return image_label_list
        
    def _load_image_gt_box(self):
        image_gt_box_list = []
        for i, index in enumerate(self._image_index):
            image_gt_box_list.append(self._load_pascal_gt_box(index, self._image_index_data_path[i]))
        return image_gt_box_list

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(self.root, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_index = []
        image_index_data_path = []
        for i, data_path in enumerate(self._data_path):
            image_set_file = os.path.join(data_path, 'ImageSets', 'Main',
                                          self._image_set + '.txt')
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                image_index_part = [x.strip() for x in f.readlines()]
                image_index.extend(image_index_part)
                image_index_data_path.extend([i]*len(image_index_part))
        return image_index, image_index_data_path

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        # return [self.root for year in self._year]
        return [os.path.join(self.root, 'VOCdevkit' + str(year)) for year in self._year]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_labels(index, self._image_index_data_path[i])
                    for i, index in enumerate(self.image_index)]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            roidb = self._load_selective_search_roidb(gt_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        for i in range(5):
            print("！！！！！！！！！加载ROI的代码还需要修改！！！！！！！！！！")

        filename = os.path.abspath(os.path.join(self.root,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_gt_box(self, index, data_path_i):
        filename = os.path.join(self._data_path[data_path_i], 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float)


        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]

        return boxes

    def _load_pascal_labels(self, index, data_path_i):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path[data_path_i], 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            gt_classes[ix] = cls

        real_label = np.zeros([1, self._num_classes], dtype=np.float32)
        for label in gt_classes:
            real_label[0, label] = 1
        return real_label.reshape(-1)
        #return {'labels' : real_label}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template_test(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = '{:s}.txt'
        filedir = ['/home/LiaoMingxiang/Workspace/weak_det/YOLOS_CAM_Test_Loc/corloc_cache_bak']
        for file in filedir:
            if not os.path.exists(file):
                os.makedirs(file)

        path = [os.path.join(file, filename) for file in filedir]
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename_list = self._get_voc_results_file_template()
            filename_list = [filename.format(cls) for filename in filename_list]
            for im_ind, index in enumerate(self._image_index):
                data_path_i = self._image_index_data_path[im_ind]
                filename = filename_list[data_path_i]
                with open(filename, 'a') as f:
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        try:
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, -1],
                                        dets[k, 0] + 1, dets[k, 1] + 1,
                                        dets[k, 2] + 1, dets[k, 3] + 1))
                        except BaseException:
                            print(f'index: {index}')
                            print(f'k: {k}')
                            print(f'dets: {dets}')
                            pdb.set_trace()

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path[0],
            'VOC' + str(self._year_test),
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path[0],
            'VOC' + str(self._year_test),
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path[0], 'annotations_cache_{}'.format(self._year_test))
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year_test) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename_list = self._get_voc_results_file_template()
            filename_list = [filename.format(cls) for filename in filename_list]
            rec, prec, ap = voc_eval(
                filename_list, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(self.cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(self.cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _eval_discovery(self, output_dir):
        annopath = [os.path.join(
            devkit_path,
            'VOC' + str(year),
            'Annotations',
            '{:s}.xml') for devkit_path, year in zip(self._devkit_path, self._year)]
        imagesetfile = [os.path.join(
            devkit_path,
            'VOC' + str(year),
            'ImageSets',
            'Main',
            self._image_set + '.txt') for devkit_path, year in zip(self._devkit_path, self._year)]
        cachedir = [os.path.join(devkit_path, 'annotations_dis_cache_{}_{}'.format(str(year), str(self._image_set))) for devkit_path, year in zip(self._devkit_path, self._year)]
        corlocs = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename_list = self._get_voc_results_file_template()
            filename_list = [filename.format(cls) for filename in filename_list]
            corloc = dis_eval(
                filename_list, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            corlocs += [corloc]
            print('CorLoc for {} = {:.4f}'.format(cls, corloc))
            with open(os.path.join(output_dir, cls + '_corloc.pkl'), 'wb') as f:
                pickle.dump({'corloc': corloc}, f)
        print('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
        print('~~~~~~~~')
        print('Results:')
        for corloc in corlocs:
            print('{:.3f}'.format(corloc))
        print('{:.3f}'.format(np.mean(corlocs)))
        print('~~~~~~~~')
        return np.mean(corlocs)

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename =self._get_comp_id() + '_det_' + self._image_set +  '_{:s}.txt'
        filedir = [os.path.join(devkit_path, 'results', 'VOC' + str(year), 'Main') for devkit_path, year in zip(self._devkit_path, self._year)]
        for file in filedir:
            if not os.path.exists(file):
                os.makedirs(file)

        path = [os.path.join(file, filename) for file in filedir]
        return path

    def _eval_discovery_test(self, output_dir):
        annopath = [os.path.join(
            devkit_path,
            'VOC' + str(year),
            'Annotations',
            '{:s}.xml') for devkit_path, year in zip(self._devkit_path, self._year)]
        imagesetfile = [os.path.join(
            devkit_path,
            'VOC' + str(year),
            'ImageSets',
            'Main',
            self._image_set + '.txt') for devkit_path, year in zip(self._devkit_path, self._year)]
        cachedir = [os.path.join(devkit_path, 'annotations_dis_cache_{}_{}'.format(str(year), str(self._image_set))) for devkit_path, year in zip(self._devkit_path, self._year)]
        corlocs = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename_list = self._get_voc_results_file_template_test()
            print('filename_list:', filename_list)
            filename_list = [filename.format(cls) for filename in filename_list]
            corloc = dis_eval(
                filename_list, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            corlocs += [corloc]
            print('CorLoc for {} = {:.4f}'.format(cls, corloc))
            with open(os.path.join(output_dir, cls + '_corloc.pkl'), 'wb') as f:
                pickle.dump({'corloc': corloc}, f)
        print('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
        print('~~~~~~~~')
        print('Results:')
        for corloc in corlocs:
            print('{:.3f}'.format(corloc))
        print('{:.3f}'.format(np.mean(corlocs)))
        print('~~~~~~~~')
        return np.mean(corlocs)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename_list = self._get_voc_results_file_template()
                filename_list = [filename.format(cls) for filename in filename_list]
                for filename in filename_list:
                    os.remove(filename)

    def evaluate_discovery(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        corlocs = self._eval_discovery(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename_list = self._get_voc_results_file_template()
                filename_list = [filename.format(cls) for filename in filename_list]
                for filename in filename_list:
                    os.remove(filename)
        return corlocs

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


def build(image_set, args):
    is_train = image_set == 'train'
    dataset = VOCDataset(args.test_path, args, is_train)
    args.num_classes = 20
    return dataset
