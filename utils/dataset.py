import os
import numpy as np
import tensorflow as tf
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

from coco.coco import *

coco_num_classes = 81
 
coco_class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush', 80: 'background'}

coco_class_colors = [[ 37, 160, 100], [159, 174, 143], [ 93, 211,  56], [119, 232, 183], [148, 118, 118], [224,  57, 247],  [107, 238, 198], [199, 249, 218], [149,  79, 175], [ 68, 173, 241], [199, 216, 169], [187, 221, 153], [140, 214, 174], [187, 123, 151], [100,  47, 166], [148, 149, 137], [143, 191, 108], [213,  81, 118], [ 73, 240, 116], [211, 241, 187],  [142,  83, 112], [167, 199, 201], [110, 125,  35], [ 69,  70, 175], [ 69,  31,  53], [236, 116,  74], [151,  88, 212],  [201,  32, 131], [ 91,  80,  75], [247, 239,  51], [ 42, 176, 184], [155,  81, 219], [101,  55, 176], [227,  35, 190],  [212, 182, 167], [166, 174, 134], [221, 101, 125], [115,  57, 246], [206, 143, 236], [146, 155, 229], [247,  35, 209],  [118, 104, 243], [ 59,  75, 239], [ 37,  41,  89], [207, 159, 130], [240,  43,  78], [219,  66, 126], [129, 246, 171],  [ 76, 250,  61], [ 46,  73, 171], [210, 233,  68], [230, 178, 241], [185,  48,  61], [ 88, 206,  60], [214, 218,  45],  [254, 247,  54], [172, 120, 182], [ 66, 227,  85], [ 95, 202, 105], [109, 103,  43], [ 87,  63, 162], [ 34, 142,  63],  [174, 215, 244], [ 56, 131, 134], [181, 213, 142], [166,  60,  40], [ 82,  71, 213], [ 72,  89,  70], [240,  76, 143],  [ 96, 168, 232], [ 36,  77, 234], [ 77, 106, 194], [184, 113, 154], [188, 210, 220], [ 85, 243,  80], [ 88, 201,  49],  [ 52, 254, 180], [196, 107,  75], [ 83, 234,  70], [ 53, 208, 181], [124, 211,  84]]

coco_class_to_category = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90, 80: 100}

coco_category_to_class = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 100: 80}


pascal_num_classes = 21

pascal_class_names = {0: 'person', 1: 'bird', 2: 'cat', 3: 'cow', 4: 'dog', 5: 'horse', 6: 'sheep', 7: 'aeroplane', 8: 'bicycle', 9: 'boat', 10: 'bus', 11: 'car', 12: 'motorbike', 13: 'train', 14: 'bottle', 15: 'chair', 16: 'diningtable', 17: 'pottedplant', 18: 'sofa', 19: 'tvmonitor', 20: 'background'}

pascal_class_ids = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19, 'background': 20} 

pascal_class_colors = [[129, 116,  94], [ 36, 143, 119], [123, 123,  68], [212, 101,  36], [ 56,  75, 190],  [ 37, 214, 129], [116,  37,  82], [ 41, 145, 158], [163, 222, 238], [205,  46,  68], [172, 181,  36], [228, 103, 189], [ 53, 175, 241], [ 94, 167,  99], [ 96,  68, 133], [245, 198, 203], [ 56,  54, 247], [170,  50, 248], [ 34,  36,  65], [206,  81, 143], [ 39, 246,  49]]


class DataSet():
    def __init__(self, img_ids, img_files, img_heights, img_widths, batch_size=1, anchor_files=None, roi_files=None, gt_classes=None, gt_bboxes=None, is_train=False, shuffle=False):
        self.img_ids = np.array(img_ids)
        self.img_files = np.array(img_files)
        self.img_heights = np.array(img_heights)
        self.img_widths = np.array(img_widths)
        self.anchor_files = np.array(anchor_files)
        self.roi_files = np.array(roi_files)
        self.batch_size = batch_size
        self.gt_classes = gt_classes
        self.gt_bboxes = gt_bboxes
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        self.current_index = 0
        self.count = len(self.img_files)
        self.indices = list(range(self.count))
        self.num_batches = int(self.count/self.batch_size)
        self.reset()

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        self.reset()

    def set_is_train(self, is_train):
        self.is_train = is_train

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.setup()

    def next_batch_for_rpn(self):
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        img_files = self.img_files[current_indices]
        if self.is_train:       
            anchor_files = self.anchor_files[current_indices]
            self.current_index += self.batch_size
            return img_files, anchor_files
        else:
            self.current_index += self.batch_size
            return img_files

    def next_batch_for_classifier(self):
        assert self.has_next_batch()       
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        img_files = self.img_files[current_indices]
        if self.is_train:       
            roi_files = self.roi_files[current_indices]
            self.current_index += self.batch_size
            return img_files, roi_files
        else:
            self.current_index += self.batch_size
            return img_files

    def next_batch_for_all(self):
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        img_files = self.img_files[current_indices]
        if self.is_train:       
            anchor_files = self.anchor_files[current_indices]
            roi_files = self.roi_files[current_indices]
            self.current_index += self.batch_size
            return img_files, anchor_files, roi_files
        else:
            self.current_index += self.batch_size
            return img_files

    def has_next_batch(self):
        return self.current_index + self.batch_size <= self.count


def prepare_train_coco_data(args):
    image_dir, annotation_file, data_dir = args.train_coco_image_dir, args.train_coco_annotation_file, args.train_coco_data_dir
    batch_size = args.batch_size
    basic_model = args.basic_model
    num_rois = args.num_rois

    coco = COCO(annotation_file)

    img_ids = list(coco.imgToAnns.keys())
    img_files = []
    img_heights = []
    img_widths = []
    anchor_files = []
    roi_files = []
    gt_classes = []
    gt_bboxes = []

    for img_id in img_ids:
        img_files.append(os.path.join(image_dir, coco.imgs[img_id]['file_name'])) 
        img_heights.append(coco.imgs[img_id]['height']) 
        img_widths.append(coco.imgs[img_id]['width']) 
        anchor_files.append(os.path.join(data_dir, os.path.splitext(coco.imgs[img_id]['file_name'])[0]+'_'+basic_model+'_anchor.npz')) 
        roi_files.append(os.path.join(data_dir, os.path.splitext(coco.imgs[img_id]['file_name'])[0]+'_'+basic_model+'_'+str(num_rois)+'_roi.npz')) 

        classes = [] 
        bboxes = [] 
        for ann in coco.imgToAnns[img_id]: 
            classes.append(coco_category_to_class[ann['category_id']]) 
            bboxes.append([ann['bbox'][1], ann['bbox'][0], ann['bbox'][3]+1, ann['bbox'][2]+1]) 

        gt_classes.append(classes)  
        gt_bboxes.append(bboxes) 
 
    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, roi_files, gt_classes, gt_bboxes, True, True)
    print("Dataset built.")
    return coco, dataset


def prepare_train_pascal_data(args):
    image_dir, annotation_dir, data_dir = args.train_pascal_image_dir, args.train_pascal_annotation_dir, args.train_pascal_data_dir
    batch_size = args.batch_size
    basic_model = args.basic_model
    num_rois = args.num_rois

    files = os.listdir(annotation_dir)
    img_ids = list(range(len(files)))

    img_files = []
    img_heights = []
    img_widths = []
    anchor_files = []
    roi_files = []
    gt_classes = []
    gt_bboxes = []

    for f in files:
        annotation = os.path.join(annotation_dir, f)

        tree = ET.parse(annotation)
        root = tree.getroot()

        img_name = root.find('filename').text 
        img_file = os.path.join(image_dir, img_name)
        img_files.append(img_file) 

        img_id_str = os.path.splitext(img_name)[0]

        size = root.find('size')
        img_height = int(size.find('height').text)
        img_width = int(size.find('width').text)
        img_heights.append(img_height) 
        img_widths.append(img_width) 

        anchor_files.append(os.path.join(data_dir, img_id_str+'_'+basic_model+'_anchor.npz')) 
        roi_files.append(os.path.join(data_dir, img_id_str+'_'+basic_model+'_'+str(num_rois)+'_roi.npz')) 

        classes = [] 
        bboxes = [] 
        for obj in root.findall('object'): 
            class_name = obj.find('name').text
            class_id = pascal_class_ids[class_name]
            classes.append(class_id) 

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bboxes.append([ymin, xmin, ymax-ymin+1, xmax-xmin+1]) 

        gt_classes.append(classes)  
        gt_bboxes.append(bboxes) 
 
    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, roi_files, gt_classes, gt_bboxes, True, True)
    print("Dataset built.")
    return dataset


def prepare_val_coco_data(args):
    image_dir, annotation_file = args.val_coco_image_dir, args.val_coco_annotation_file

    coco = COCO(annotation_file)

    img_ids = list(coco.imgToAnns.keys())
    img_files = []
    img_heights = []
    img_widths = []

    for img_id in img_ids:
        img_files.append(os.path.join(image_dir, coco.imgs[img_id]['file_name']))
        img_heights.append(coco.imgs[img_id]['height'])         
        img_widths.append(coco.imgs[img_id]['width'])         

    print("Building the validation dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return coco, dataset


def prepare_val_pascal_data(args):
    image_dir, annotation_dir = args.val_pascal_image_dir, args.val_pascal_annotation_dir

    files = os.listdir(annotation_dir)
    img_ids = list(range(len(files)))

    img_files = []
    img_heights = []
    img_widths = []

    pascal = {}

    for f in files:
        annotation = os.path.join(annotation_dir, f)

        tree = ET.parse(annotation)
        root = tree.getroot()

        img_name = root.find('filename').text 
        pascal[img_name] = []

        img_file = os.path.join(image_dir, img_name)
        img_files.append(img_file) 
 
        size = root.find('size')
        img_height = int(size.find('height').text)
        img_width = int(size.find('width').text)
        img_heights.append(img_height) 
        img_widths.append(img_width) 

        for obj in root.findall('object'): 
            class_name = obj.find('name').text
            class_id = pascal_class_ids[class_name]
            temp = obj.find('difficult')
            difficult = int(temp.text) if temp!=None else 0

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            pascal[img_name].append({'class_id': class_id, 'bbox':[xmin, ymin, xmax, ymax], 'difficult': difficult})

    print("Building the validation dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return pascal, dataset


def eval_pascal_one_class(pascal, detections, c):
    gts = {} 
    num_objs = 0 
    for img_name in pascal:
        gts[img_name] = []
        for obj in pascal[img_name]:
            if obj['class_id'] == c and obj['difficult']==0:
                gts[img_name] += [{'bbox':obj['bbox'], 'detected': False}]
                num_objs += 1

    dts = []
    scores = []
    num_dets = 0
    for img_name in detections:
        for dt in detections[img_name]:
            if dt['class_id'] == c:
                dts.append([img_name, dt['bbox'], dt['score']])
                scores.append(dt['score'])
                num_dets += 1

    scores = np.array(scores, np.float32)
    sorted_idx = np.argsort(scores)[::-1]

    tp = np.zeros((num_dets))
    fp = np.zeros((num_dets))

    for i in tqdm(list(range(num_dets))):
        idx = sorted_idx[i]

        img_name = dts[idx][0]
        bbox = dts[idx][1]   
    
        gt_bboxes = np.array([obj['bbox'] for obj in gts[img_name]], np.float32)

        max_iou = 0.0

        if gt_bboxes.size > 0:
            ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
            iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
            ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
            iymax = np.minimum(gt_bboxes[:, 3], bbox[3])

            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)

            area_intersect = iw * ih

            area_union = (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0) + (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.0) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.0) - area_intersect

            ious = area_intersect / area_union
            max_iou = np.max(ious, axis=0)
            j = np.argmax(ious)

        if max_iou > 0.5:
            if not gts[img_name][j]['detected']:
                tp[i] = 1.0
                gts[img_name][j]['detected'] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp * 1.0 / num_objs
    prec = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    print('average precision for class %s = %f' %(pascal_class_names[c], ap))

    return ap


def eval_pascal(pascal, detections):
    ap = 0.0 
    for i in range(pascal_num_classes-1):
        ap += eval_pascal_one_class(pascal, detections, i)
    ap = ap / (pascal_num_classes-1)
    print('mean average precision = %f' %ap)
    return ap


def prepare_test_data(args):
    image_dir = args.test_image_dir

    files = os.listdir(image_dir)
    files = [f for f in files if f.lower().endswith('.jpg')]

    img_ids = list(range(len(files)))
    img_files = []
    img_heights = []
    img_widths = []
      
    for f in files:
        img_path = os.path.join(image_dir, f)
        img_files.append(img_path)
        img = cv2.imread(img_path)
        img_heights.append(img.shape[0]) 
        img_widths.append(img.shape[1]) 

    print("Building the testing dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return dataset

