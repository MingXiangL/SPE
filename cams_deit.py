import os
import cv2
import numpy as np
import pickle
import torch
import pdb
import math

def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam , (size[0], size[1]))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def blend_cam(image, cam):
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.5 + heatmap * 0.5

    return blend, heatmap

def blend_cam_list(image, cam_list):
    cam_list = [(cam * 255.).astype(np.uint8) for cam in cam_list]
    heatmap_list = [cv2.applyColorMap(cam, cv2.COLORMAP_JET) for cam in cam_list]
    alpha = 1./ (1+len(heatmap_list))
    blend = alpha * image
    for heatmap in heatmap_list:
        blend = blend + alpha * heatmap

    return blend


def get_bboxes(cam, cam_thr=0.2):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)


def get_multi_bboxes(cam, cam_thr=0.2, area_ratio=0.5):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
                                       
    if len(contours) != 0:
        estimated_bbox = []
        areas = list(map(cv2.contourArea, contours))
        area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
        for idx in area_idx:
            if areas[idx] >= areas[area_idx[0]] * area_ratio:
                c = contours[idx]
                x, y, w, h = cv2.boundingRect(c)
                estimated_bbox.append([x, y, x + w, y + h])
        # areas1 = sorted(areas, reverse=True)
        
        # pdb.set_trace()
        
        # estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [[0, 0, 1, 1]]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)

def get_multi_bboxes_with_area(cam, cam_thr=0.2, area_ratio=0.5):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
                                       
    if len(contours) != 0:
        estimated_bbox = []
        areas_ret = []
        areas = list(map(cv2.contourArea, contours))
        area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
        for idx in area_idx:
            if areas[idx] >= areas[area_idx[0]] * area_ratio:
                c = contours[idx]
                x, y, w, h = cv2.boundingRect(c)
                estimated_bbox.append([x, y, x + w, y + h])
            areas_ret.append(areas[idx])
        # areas1 = sorted(areas, reverse=True)
        
        # pdb.set_trace()
        
        # estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [[0, 0, 1, 1]]
        areas_ret = [1]

    return estimated_bbox, areas_ret  #, thr_gray_heatmap, len(contours)

def bgrtensor2image(input, image_mean, image_std):
    image_mean = torch.reshape(torch.tensor(image_mean), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(image_std), (1, 3, 1, 1))
    image = input * image_mean + image_std
    image = image.numpy().transpose(0, 2, 3, 1)
    image = image[:, :, :, ::-1] * 255
    #image = image * 255
    return image

def tensor2image(input, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):
    image_mean = torch.reshape(torch.tensor(image_mean), (1, 3, 1, 1)).to(input)
    image_std = torch.reshape(torch.tensor(image_std), (1, 3, 1, 1)).to(input)
    image = input * image_mean + image_std
    image = image.cpu().numpy().transpose(0, 2, 3, 1)
    image = image[:, :, :, ::-1] * 255
    return image


def calculate_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def draw_wogt_bbox(image, pred_box_list, gt_score_list):

    def draw_bbox(img, box_list, gt_score_list, color1=(0, 0, 255)):
        for i in range(len(box_list)):
            #print(box_list)
            #print(box_list[i])
            #print(box_list[i,0], box_list[i,1], box_list[i,2], box_list[i,3])
            cv2.rectangle(img, (box_list[i][0], box_list[i][1]), (box_list[i][2], box_list[i][3]), color1, 2)
            cv2.putText(img, '%.1f' % (gt_score_list[i]), (box_list[i][0], box_list[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, pred_box_list, gt_score_list)

    # mark the iou
    #mark_target(boxed_image, '%.1f' % (gt_score * 100), (140, 30), 2)
    # mark_target(boxed_image, 'IOU%.2f' % (iou), (80, 30), 2)
    # # mark the top1
    # if is_top1:
    #     mark_target(boxed_image, 'Top1', (10, 30))
    # mark_target(boxed_image, 'GT_Score%.2f' % (gt_score), (10, 200), 2)

    return boxed_image

def draw_gt_bbox(image, gt_box_list, pred_box_list, gt_score_list):

    def draw_bbox(img, gt_box_list, box_list, gt_score_list, color1=(0, 0, 255), color2=(0, 255, 0)):
        
        for i in range(len(box_list)):
            #print(box_list)
            #print(box_list[i])
            #print(box_list[i,0], box_list[i,1], box_list[i,2], box_list[i,3])
            cv2.rectangle(img, (box_list[i][0], box_list[i][1]), (box_list[i][2], box_list[i][3]), color1, 2)
            cv2.putText(img, '%.1f' % (100*gt_score_list[i]), (box_list[i][0], box_list[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for i in range(len(gt_box_list)):
            cv2.rectangle(img, (int(gt_box_list[i,0]), int(gt_box_list[i,1])), \
                (int(gt_box_list[i,2]), int(gt_box_list[i,3])), color2, 2)

        return img

    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, gt_box_list, pred_box_list, gt_score_list)

    return boxed_image

def draw_bbox(image, iou, gt_box, pred_box, gt_score, is_top1=False):

    def draw_bbox(img, box1, box2, color1=(0, 0, 255), color2=(0, 255, 0)):
        for i in range(len(box1)):
            cv2.rectangle(img, (box1[i,0], box1[i,1]), (box1[i,2], box1[i,3]), color1, 2)
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color2, 2)
        return img

    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, gt_box, pred_box)

    # mark the iou
    mark_target(boxed_image, '%.1f' % (iou * 100), (140, 30), 2)
    # mark_target(boxed_image, 'IOU%.2f' % (iou), (80, 30), 2)
    # # mark the top1
    # if is_top1:
    #     mark_target(boxed_image, 'Top1', (10, 30))
    # mark_target(boxed_image, 'GT_Score%.2f' % (gt_score), (10, 200), 2)

    return boxed_image

def evaluate_cls_loc(input, cls_label, bbox_label, logits, cams, image_names, cfg, epoch):
    """
    :param input: input tensors of the model
    :param cls_label: class label
    :param bbox_label: bounding box label
    :param logits: classification scores
    :param cams: cam of all the classes
    :param image_names: names of images
    :param cfg: configurations
    :param epoch: epoch
    :return: evaluate results
    """
    cls_top1 = []
    cls_top5 = []
    loc_top1 = []
    loc_top5 = []
    loc_gt_known = []
    top1_loc_right = []
    top1_loc_cls = []
    top1_loc_mins = []
    top1_loc_part = []
    top1_loc_more = []
    top1_loc_wrong = []

    # label, top1 and top5 results
    cls_label = cls_label.tolist()
    cls_scores = logits.tolist()
    _, top1_idx = logits.topk(1, 1, True, True)
    top1_idx = top1_idx.tolist()
    _, top5_idx = logits.topk(5, 1, True, True)
    top5_idx = top5_idx.tolist()

    k = cfg.MODEL.TOP_K
    _, topk_idx = logits.topk(k, 1, True, True)
    topk_idx = topk_idx.tolist()

    batch = cams.shape[0]
    image = tensor2image(input.clone().detach().cpu(), cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)

    for b in range(batch):
        gt_bbox = bbox_label[b].strip().split(' ')
        gt_bbox = list(map(float, gt_bbox))
        top_bboxes, top_mask=get_topk_boxes(top5_idx[b], cams[b], cfg.DATA.CROP_SIZE, threshold=cfg.MODEL.CAM_THR)
        topk_cls, topk_loc, wrong_details=cls_loc_err(top_bboxes, cls_label[b], gt_bbox, topk=(1,5))
        cls_top1_b, cls_top5_b = topk_cls
        loc_top1_b, loc_top5_b = topk_loc
        cls_top1.append(cls_top1_b)
        cls_top5.append(cls_top5_b)
        loc_top1.append(loc_top1_b)
        loc_top5.append(loc_top5_b)
        cls_wrong, multi_instances, region_part, region_more, region_wrong = wrong_details
        right = 1 - (cls_wrong + multi_instances + region_part + region_more + region_wrong)
        top1_loc_right.append(right)
        top1_loc_cls.append(cls_wrong)
        top1_loc_mins.append(multi_instances)
        top1_loc_part.append(region_part)
        top1_loc_more.append(region_more)
        top1_loc_wrong.append(region_wrong)
        # gt_known
        # mean top k
        print(f'cams.size():{cams.size()}')
        cam_b = cams[b, [cls_label[b]], :, :]
        cam_b = torch.mean(cam_b, dim=0, keepdim=True)

        cam_b = cam_b.detach().cpu().numpy().transpose(1, 2, 0)
        # Resize and Normalize CAM
        cam_b = resize_cam(cam_b, size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))

        # Estimate BBOX
        estimated_bbox = get_bboxes(cam_b, cam_thr=cfg.MODEL.CAM_THR)

        # Calculate IoU
        gt_box_cnt = len(gt_bbox) // 4
        max_iou = 0
        for i in range(gt_box_cnt):
            gt_box = gt_bbox[i * 4:(i + 1) * 4]
            iou_i = cal_iou(estimated_bbox, gt_box)
            if iou_i > max_iou:
                max_iou = iou_i

        iou = max_iou
        # iou = calculate_IOU(bbox_label[b].numpy(), estimated_bbox)

        # print('cam_b shape', cam_b.shape, 'cam_b max', cam_b.max(), 'cam_b min', cam_b.min(), 'thre', cfg.MODEL.CAM_THR, 'iou ', iou)
        #if iou < 0.5:
        #    pdb.set_trace()
        # gt known
        if iou >= 0.5:
            loc_gt_known.append(1)
        else:
            loc_gt_known.append(0)

        # Get blended image
        blend, heatmap = blend_cam(image[b], cam_b)
        # Get boxed image
        gt_score = cls_scores[b][top1_idx[b][0]]  # score of gt class
        boxed_image = draw_bbox(blend, iou, np.array(gt_bbox).reshape(-1,4).astype(np.int), estimated_bbox, gt_score, False)

        # save result
        if cfg.TEST.SAVE_BOXED_IMAGE:
            image_name = image_names[b]

            save_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'boxed_image', str(epoch), image_name.split('/')[0])
            save_path = os.path.join(cfg.BASIC.SAVE_DIR, 'boxed_image', str(epoch), image_name)
            os.makedirs(save_dir, exist_ok=True)
            # print(save_path)
            cv2.imwrite(save_path, boxed_image)

    return cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known, top1_loc_right, top1_loc_cls, top1_loc_mins, \
           top1_loc_part, top1_loc_more, top1_loc_wrong


def get_topk_boxes(cls_inds, cam_map, crop_size, topk=(1, 5), threshold=0.2, ):
    maxk_boxes = []
    maxk_maps = []
    for cls in cls_inds:
        cam_map_ = cam_map[[cls], :, :]
        cam_map_ = cam_map_.detach().cpu().numpy().transpose(1, 2, 0)
        # Resize and Normalize CAM
        cam_map_ = resize_cam(cam_map_, size=(crop_size, crop_size))
        maxk_maps.append(cam_map_.copy())

        # Estimate BBOX
        estimated_bbox = get_bboxes(cam_map_, cam_thr=threshold)
        maxk_boxes.append([cls] + estimated_bbox)

    result = [maxk_boxes[:k] for k in topk]

    return result, maxk_maps

def cls_loc_err(topk_boxes, gt_label, gt_boxes, topk=(1,), iou_th=0.5):
    assert len(topk_boxes) == len(topk)
    gt_boxes = gt_boxes
    gt_box_cnt = len(gt_boxes) // 4
    topk_loc = []
    topk_cls = []
    for topk_box in topk_boxes:
        loc_acc = 0
        cls_acc = 0
        for cls_box in topk_box:
            max_iou = 0
            max_gt_id = 0
            for i in range(gt_box_cnt):
                gt_box = gt_boxes[i*4:(i+1)*4]
                iou_i = cal_iou(cls_box[1:], gt_box)
                if  iou_i> max_iou:
                    max_iou = iou_i
                    max_gt_id = i
            if len(topk_box)  == 1:
                wrong_details = get_badcase_detail(cls_box, gt_boxes, gt_label, max_iou, max_gt_id)
            if cls_box[0] == gt_label:
                cls_acc = 1
            if cls_box[0] == gt_label and max_iou > iou_th:
                loc_acc = 1
                break
        topk_loc.append(float(loc_acc))
        topk_cls.append(float(cls_acc))
    return topk_cls, topk_loc, wrong_details


def cal_iou(box1, box2, method='iou'):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    if method == 'iog':
        iou_val = i_area / (box2_area)
    elif method == 'iob':
        iou_val = i_area / (box1_area)
    else:
        iou_val = i_area / (box1_area + box2_area - i_area)
    return iou_val

def get_badcase_detail(top1_bbox, gt_bboxes, gt_label, max_iou, max_gt_id):
    cls_wrong = 0
    multi_instances = 0
    region_part = 0
    region_more = 0
    region_wrong = 0

    pred_cls = top1_bbox[0]
    pred_bbox = top1_bbox[1:]

    if not int(pred_cls) == gt_label:
        cls_wrong = 1
        return cls_wrong, multi_instances, region_part, region_more, region_wrong

    if max_iou > 0.5:
        return 0, 0, 0, 0, 0

    # multi_instances error
    gt_box_cnt = len(gt_bboxes) // 4
    if gt_box_cnt > 1:
        iogs = []
        for i in range(gt_box_cnt):
            gt_box = gt_bboxes[i * 4:(i + 1) * 4]
            iog = cal_iou(pred_bbox, gt_box, method='iog')
            iogs.append(iog)
        if sum(np.array(iogs) > 0.3)> 1:
            multi_instances = 1
            return cls_wrong, multi_instances, region_part, region_more, region_wrong
    # region part error
    iog = cal_iou(pred_bbox, gt_bboxes[max_gt_id*4:(max_gt_id+1)*4], method='iog')
    iob = cal_iou(pred_bbox, gt_bboxes[max_gt_id*4:(max_gt_id+1)*4], method='iob')
    if iob >0.5:
        region_part = 1
        return cls_wrong, multi_instances, region_part, region_more, region_wrong
    if iog >= 0.7:
        region_more = 1
        return cls_wrong, multi_instances, region_part, region_more, region_wrong
    region_wrong = 1
    return cls_wrong, multi_instances, region_part, region_more, region_wrong

class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i
