#!/usr/bin/env python
import os
import sys
import argparse
import tensorflow as tf

from model import *
from utils.dataset import *
from utils.coco.coco import *

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--phase', default = 'train')
    parser.add_argument('--component_to_train', default = 'all')
    parser.add_argument('--load', action = 'store_true', default = False)

    parser.add_argument('--mean_file', default = './utils/ilsvrc_2012_mean.npy')
    parser.add_argument('--basic_model', default = 'vgg16')
    parser.add_argument('--basic_model_file', default = './tfmodels/vgg16.tfmodel')
    parser.add_argument('--load_basic_model', action = 'store_true', default = False)

    parser.add_argument('--dataset', default = 'pascal')

    parser.add_argument('--train_coco_image_dir', default = './train/coco/images/')
    parser.add_argument('--train_coco_annotation_file', default = './train/coco/instances_train2014.json') 
    parser.add_argument('--train_coco_data_dir', default = './train/coco/data/')

    parser.add_argument('--train_pascal_image_dir', default = './train/pascal/images/')
    parser.add_argument('--train_pascal_annotation_dir', default = './train/pascal/annotations/') 
    parser.add_argument('--train_pascal_data_dir', default = './train/pascal/data/')

    parser.add_argument('--val_coco_image_dir', default = './val/coco/images/')
    parser.add_argument('--val_coco_annotation_file', default = './val/coco/instances_val2014.json')

    parser.add_argument('--val_pascal_image_dir', default = './val/pascal/images/')
    parser.add_argument('--val_pascal_annotation_dir', default = './val/pascal/annotations/')

    parser.add_argument('--test_image_dir', default = './test/images/')
    parser.add_argument('--test_result_file', default = './test/result.pickle')
    parser.add_argument('--test_result_dir', default = './test/results/')

    parser.add_argument('--save_dir', default = './models/')
    parser.add_argument('--save_period', type = int, default = 1)
    
    parser.add_argument('--solver', default = 'adam') 
    parser.add_argument('--num_epochs', type = int, default = 100) 
    parser.add_argument('--batch_size', type = int, default = 16) 
    parser.add_argument('--learning_rate', type = float, default = 2e-5) 
    parser.add_argument('--momentum', type = float, default = 0.9) 
    parser.add_argument('--decay', type = float, default = 0.9) 
    parser.add_argument('--weight_decay', type = float, default = 3e-4)    
    parser.add_argument('--batch_norm', action = 'store_true', default = False) 
    parser.add_argument('--rpn_weight', type = float, default = 1.0)    
    parser.add_argument('--cls_weight', type = float, default = 1.0)    
    parser.add_argument('--rpn_reg_weight', type = float, default = 10.0)    
    parser.add_argument('--cls_reg_weight', type = float, default = 10.0)    
    
    parser.add_argument('--num_rois', type = int, default = 100)    
    parser.add_argument('--num_object_per_class', type = int, default = 10)    
    parser.add_argument('--bbox_reg', action = 'store_true', default = False)    
    parser.add_argument('--bbox_per_class', action = 'store_true', default = False)    

    args = parser.parse_args()

    with tf.Session() as sess:
        if args.phase == 'train':
            if args.dataset == 'coco':
                train_coco, train_data = prepare_train_coco_data(args)
            else:
                train_data = prepare_train_pascal_data(args)

            model = ObjectDetector(args, 'train')
            sess.run(tf.initialize_all_variables())

            if args.load:
                model.load(sess)
            elif args.load_basic_model:
                model.load2(args.basic_model_file, sess)

            if args.component_to_train == 'all':               # train everything
                model.train(sess, train_data)
            elif args.component_to_train == 'rpn':             # train rpn only
                model.train_rpn(sess, train_data)
            else:
                model.train_classifier(sess, train_data)       # train classifier only
 
        elif args.phase == 'val':
            model = ObjectDetector(args, 'val')
            model.load(sess)

            if args.dataset == 'coco':
                val_coco, val_data = prepare_val_coco_data(args)
                model.val_coco(sess, val_coco, val_data)
            else:
                val_pascal, val_data = prepare_val_pascal_data(args)
                model.val_pascal(sess, val_pascal, val_data)

        else:
            test_data = prepare_test_data(args)
            model = ObjectDetector(args, 'test')  
            model.load(sess)
            model.test(sess, test_data)

if __name__=="__main__":
     main(sys.argv)

