import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import pdb

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def dis_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.3):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    for cachedir_i in cachedir:
        if not os.path.isdir(cachedir_i):
            os.mkdir(cachedir_i)
    cachefile = [os.path.join(cachedir_i, 'annots.pkl') for cachedir_i in cachedir]
    # read list of images
    lines_list = []
    imagenames_list = []
    for imagesetfile_i in imagesetfile:
        with open(imagesetfile_i, 'r') as f:
            lines = f.readlines()
            lines_list.append(lines)
    for lines in lines_list:
        imagenames = [x.strip() for x in lines]
        imagenames_list.append(imagenames)
    recs = {}
    for i_data_path, cachefile_i in enumerate(cachefile):
        #print(cachefile_i)
        if not os.path.isfile(cachefile_i):
            # load annots
            imagenames = imagenames_list[i_data_path]
            recs_i = {}
            for i, imagename in enumerate(imagenames):
                recs_i[imagename] = parse_rec(annopath[i_data_path].format(imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile_i))
            with open(cachefile_i, 'wb') as f:
                pickle.dump(recs_i, f)
            recs.update(recs_i)
        else:
            # load
            with open(cachefile_i, 'rb') as f:
                recs_i = pickle.load(f)
                recs.update(recs_i)
                #print(cachefile_i)
                #print(len(recs_i))
    #print('len(recs)', len(recs))
    imagenames = []
    for e in imagenames_list:
        imagenames.extend(e)
    #print('len(imagenames)', len(imagenames))
    # extract gt objects for this class
    class_recs = {}
    nimgs = 0.0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        nimgs = nimgs + float(bbox.size > 0)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets
    detfile = [detpath_i.format(classname) for detpath_i in detpath]
    lines = []
    for detfile_i in detfile:
        with open(detfile_i, 'r') as f:
            lines_i = f.readlines()
            lines.extend(lines_i)

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            tp[d] = 1.
            continue
        # else:
        #     print(f'missed image id:{image_ids[d]}, iou:{ovmax}')
    return np.sum(tp) / nimgs