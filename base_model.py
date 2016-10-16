import os
import math
import sys
import numpy as np
import tensorflow as tf
import cv2
import cPickle as pickle
from tqdm import tqdm

from utils.bbox import *
from utils.dataset import *
from utils.coco.coco import *
from utils.coco.cocoeval import *

class ImageLoader(object):
    def __init__(self, mean_file):
        self.isotropic = False 
        self.channels = 3
        self.bgr = True
        self.scale_shape = [640, 640]
        self.crop_shape = [640, 640]
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_img(self, image_file):
        file_data = tf.read_file(image_file)

        img = tf.image.decode_jpeg(file_data, channels=self.channels)
        img = tf.reverse(img, [False, False, self.bgr]) 

        if self.isotropic:
            img_shape = tf.to_float(tf.shape(img)[:2])
            min_length = tf.minimum(img_shape[0], img_shape[1])
            scale_shape = tf.pack(self.scale_shape)
            new_shape = tf.to_int32((scale_shape / min_length) * img_shape)
        else:
            new_shape = tf.pack(self.scale_shape)

        img = tf.image.resize_images(img, new_shape[0], new_shape[1])

        crop_shape = tf.pack(self.crop_shape)
        offset = (new_shape - crop_shape) / 2     
        img = tf.slice(img, tf.to_int32([offset[0], offset[1], 0]), tf.to_int32([crop_shape[0], crop_shape[1], -1]))

        img = tf.to_float(img)-self.mean
        return img


class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params

        self.mode = mode
        self.batch_size = params.batch_size if mode=='train' else 1
        self.batch_norm = params.batch_norm

        if params.dataset == 'coco':
            self.type = 'coco'
            self.num_classes = coco_num_classes
            self.class_names = coco_class_names
            self.class_colors = coco_class_colors
            self.class_to_category = coco_class_to_category
            self.category_to_class = coco_category_to_class
        else:
            self.type = 'pascal'
            self.num_classes = pascal_num_classes
            self.class_names = pascal_class_names
            self.class_colors = pascal_class_colors
            self.class_ids = pascal_class_ids

        self.basic_model = params.basic_model
        self.num_rois = params.num_rois
        self.num_object_per_class = params.num_object_per_class
        self.bbox_reg = params.bbox_reg
        self.bbox_per_class = params.bbox_per_class

        self.label = self.type + '/' + self.basic_model + '/'
        self.save_dir = os.path.join(params.save_dir, self.label)

        self.img_loader = ImageLoader(params.mean_file)
        self.img_shape = [640, 640, 3]

        self.anchor_scales = [60, 120, 180, 240, 300, 400, 500]
        self.anchor_ratios = [[1.0/math.sqrt(3), math.sqrt(3)], [1.0/math.sqrt(2), math.sqrt(2)], [1.0, 1.0], [math.sqrt(2), 1.0/math.sqrt(2)], [1.0/math.sqrt(3), math.sqrt(3)]]
        self.num_anchors_per_location = len(self.anchor_scales) * len(self.anchor_ratios) 

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False) 
        self.build() 
        self.saver = tf.train.Saver(max_to_keep = 100) 

    def build(self):
        raise NotImplementedError()

    def process_rpn_result(self, probs, regs):
        raise NotImplementedError()

    def process_classifier_result(self, probs, clss, regs, rois, dataset, i):
        raise NotImplementedError()

    def get_feed_dict_for_rpn(self, batch, is_train, feats):
        raise NotImplementedError()

    def get_feed_dict_for_classifier(self, batch, is_train, feats, rois=None, masks=None):
        raise NotImplementedError()

    def get_feed_dict_for_all(self, batch, is_train, feats=None):
        raise NotImplementedError()

    def train_rpn(self, sess, train_dataset):
        print("Training RPN...")
        params = self.params

        self.prepare_data_for_rpn(train_dataset)

        for epoch_no in tqdm(list(range(params.num_epochs)), desc='epoch'): 
            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'):
                batch = train_dataset.next_batch_for_rpn()
                img_files, _ = batch
                feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                feed_dict = self.get_feed_dict_for_rpn(batch, is_train=True, feats=feats)
                _, loss0, loss1, global_step = sess.run([self.rpn_opt_op, self.rpn_loss0, self.rpn_loss1, self.global_step], feed_dict=feed_dict)
                print(" loss0=%f loss1=%f" %(loss0, loss1))

            train_dataset.reset()

            if (epoch_no+1) % params.save_period == 0:
                self.save(sess)

        print("RPN trained.")

    def train_classifier(self, sess, train_dataset):
        print("Training Classifier...")
        params = self.params

        self.prepare_data_for_rpn(train_dataset)

        for epoch_no in tqdm(list(range(params.num_epochs)), desc='epoch'): 

            self.prepare_data_for_classifier(train_dataset)

            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'): 
                batch = train_dataset.next_batch_for_classifier() 
                img_files, _ = batch
                feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                feed_dict = self.get_feed_dict_for_classifier(batch, is_train=True, feats=feats) 
                _, loss0, loss1, global_step = sess.run([self.cls_opt_op, self.cls_loss0, self.cls_loss1, self.global_step], feed_dict=feed_dict) 
                print(" loss0=%f loss1=%f" %(loss0, loss1)) 

            train_dataset.reset()

            if (epoch_no+1) % params.save_period == 0:
                self.save(sess)

        print("Classifier trained.")

    def train(self, sess, train_dataset):
        print("Training the model...")
        params = self.params

        self.prepare_data_for_rpn(train_dataset)

        for epoch_no in tqdm(list(range(params.num_epochs)), desc='epoch'): 

            self.prepare_data_for_classifier(train_dataset)

            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'): 
                batch = train_dataset.next_batch_for_all() 
                img_files, _, _ = batch
                feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                feed_dict = self.get_feed_dict_for_all(batch, is_train=True, feats=feats) 
                _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict) 
                print(" loss0=%f loss1=%f" %(loss0, loss1)) 

            train_dataset.reset()

            if (epoch_no+1) % params.save_period == 0:
                self.save(sess)

        print("Model trained.")

    def val_coco(self, sess, val_coco, val_dataset):
        print("Validating the model ...")
        num_rois = self.num_rois
        det_scores = []
        det_classes = []
        det_bboxes = []

        for k in tqdm(list(range(val_dataset.count))):
            batch = val_dataset.next_batch_for_rpn()
            img_files = batch

            feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})  

            feed_dict = self.get_feed_dict_for_rpn(batch, is_train=False, feats=feats)
            if self.bbox_reg:
                scores, regs = sess.run([self.rpn_scores, self.rpn_regs], feed_dict=feed_dict)
                rois = unparam_bbox(regs.squeeze(), self.anchors, self.img_shape[:2])
            else:
                scores = sess.run(self.rpn_scores, feed_dict=feed_dict)
                rois = self.anchors

            num_real_rois, rois = self.process_rpn_result(scores.squeeze(), rois)
            rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])

            padded_rois = np.ones((num_rois, 4), np.int32) * 3
            padded_rois[:num_real_rois] = rois
            padded_rois = np.expand_dims(padded_rois, 0)
            
            masks = np.zeros((num_rois), np.float32)
            masks[:num_real_rois] = 1.0
            masks = np.expand_dims(masks, 0)

            feed_dict = self.get_feed_dict_for_classifier(batch, is_train=False, feats=feats, rois=padded_rois, masks=masks)
            if self.bbox_reg:
                probs, clss, regs = sess.run([self.res_scores, self.res_clss, self.res_regs], feed_dict=feed_dict)
                bboxes = unparam_bbox(regs.squeeze(), padded_rois.squeeze())
                bboxes = convert_bbox(bboxes, self.conv_feat_shape[:2], self.img_shape[:2])
            else:
                probs, clss = sess.run([self.res_scores, self.res_clss], feed_dict=feed_dict)
                bboxes = padded_rois.squeeze()
                bboxes = convert_bbox(bboxes, self.conv_feat_shape[:2], self.img_shape[:2])

            num_dets, scores, classes, bboxes = self.process_classifier_result(probs.squeeze(), clss.squeeze(), bboxes, val_dataset, k)

            det_scores.append(scores)
            det_classes.append(classes)
            det_bboxes.append(bboxes)

        val_dataset.reset() 

        results = [] 
        for i in range(val_dataset.count): 
            for s, c, b in zip(det_scores[i], det_classes[i], det_bboxes[i]): 
                results.append({'image_id': val_dataset.img_ids[i], 'category_id': self.class_to_category[c], 'bbox':[b[1], b[0], b[3]-1, b[2]-1], 'score': s}) 

        res_coco = val_coco.loadRes2(results) 
        E = COCOeval(val_coco, res_coco) 
        E.evaluate() 
        E.accumulate() 
        E.summarize() 
        print("Validation complete.")

    def val_pascal(self, sess, val_pascal, val_dataset):
        print("Validating the model ...")
        num_rois = self.num_rois
        det_scores = []
        det_classes = []
        det_bboxes = []

        for k in tqdm(list(range(val_dataset.count))):
            batch = val_dataset.next_batch_for_rpn()
            img_files = batch

            feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})  

            feed_dict = self.get_feed_dict_for_rpn(batch, is_train=False, feats=feats)
            if self.bbox_reg:
                scores, regs = sess.run([self.rpn_scores, self.rpn_regs], feed_dict=feed_dict)
                rois = unparam_bbox(regs.squeeze(), self.anchors, self.img_shape[:2])
            else:
                scores = sess.run(self.rpn_scores, feed_dict=feed_dict)
                rois = self.anchors

            num_real_rois, rois = self.process_rpn_result(scores.squeeze(), rois)
            rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            
            padded_rois = np.ones((num_rois, 4), np.int32) * 3
            padded_rois[:num_real_rois] = rois
            padded_rois = np.expand_dims(padded_rois, 0)
            
            masks = np.zeros((num_rois), np.float32)
            masks[:num_real_rois] = 1.0
            masks = np.expand_dims(masks, 0)

            feed_dict = self.get_feed_dict_for_classifier(batch, is_train=False, feats=feats, rois=padded_rois, masks=masks)
            if self.bbox_reg:
                probs, clss, regs = sess.run([self.res_scores, self.res_clss, self.res_regs], feed_dict=feed_dict)
                bboxes = unparam_bbox(regs.squeeze(), padded_rois.squeeze())
                bboxes = convert_bbox(bboxes, self.conv_feat_shape[:2], self.img_shape[:2])
            else:
                probs, clss = sess.run([self.res_scores, self.res_clss], feed_dict=feed_dict)
                bboxes = padded_rois.squeeze()
                bboxes = convert_bbox(bboxes, self.conv_feat_shape[:2], self.img_shape[:2])

            num_dets, scores, classes, bboxes = self.process_classifier_result(probs.squeeze(), clss.squeeze(), bboxes, val_dataset, k)

            det_scores.append(scores)
            det_classes.append(classes)
            det_bboxes.append(bboxes)

        val_dataset.reset() 

        results = {} 
        for i in range(val_dataset.count): 
            file_name = val_dataset.img_files[i].split(os.sep)[-1]
            results[file_name] = []
            for s, c, b in zip(det_scores[i], det_classes[i], det_bboxes[i]): 
                results[file_name].append({'class_id': c, 'bbox':[b[1], b[0], b[1]+b[3]-1, b[0]+b[2]-1], 'score': s}) 

        eval_pascal(val_pascal, results)
        print("Validation complete.")

    def test(self, sess, test_dataset, show_rois=True, show_dets=True):
        print("Testing the model ...")
        num_rois = self.num_rois
        result_dir = self.params.test_result_dir
        det_scores = []
        det_classes = []
        det_bboxes = []
        font = cv2.FONT_HERSHEY_COMPLEX

        for k in tqdm(list(range(test_dataset.count))):
            batch = test_dataset.next_batch_for_rpn()
            img_files = batch
            img_file = img_files[0]
            img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]
            H, W = test_dataset.img_heights[k], test_dataset.img_widths[k]

            feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})           

            feed_dict = self.get_feed_dict_for_rpn(batch, is_train=False, feats=feats)
            if self.bbox_reg:
                scores, regs = sess.run([self.rpn_scores, self.rpn_regs], feed_dict=feed_dict)
                rois = unparam_bbox(regs.squeeze(), self.anchors, self.img_shape[:2])
            else:
                scores = sess.run(self.rpn_scores, feed_dict=feed_dict)
                rois = self.anchors

            num_real_rois, rois = self.process_rpn_result(scores.squeeze(), rois)

            scaled_rois = convert_bbox(rois, self.img_shape[:2], [H, W])
            img = cv2.imread(img_file)
            for roi in scaled_rois:                               
                y, x, h, w = roi
                cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (255,255,255), 2)

            if show_rois:
                winname = '%d rois' %num_real_rois
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 100, 100)
                cv2.waitKey()

            cv2.imwrite(os.path.join(result_dir, img_name+'_rois.jpg'), img)

            rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            
            padded_rois = np.ones((num_rois, 4), np.int32) * 3
            padded_rois[:num_real_rois] = rois
            padded_rois = np.expand_dims(padded_rois, 0)
            
            masks = np.zeros((num_rois), np.float32)
            masks[:num_real_rois] = 1.0
            masks = np.expand_dims(masks, 0)
            
            feed_dict = self.get_feed_dict_for_classifier(batch, is_train=False, feats=feats, rois=padded_rois, masks=masks)
            if self.bbox_reg:
                probs, clss, regs = sess.run([self.res_scores, self.res_clss, self.res_regs], feed_dict=feed_dict)
                bboxes = unparam_bbox(regs.squeeze(), padded_rois.squeeze())
                bboxes = convert_bbox(bboxes, self.conv_feat_shape[:2], self.img_shape[:2])
            else:
                probs, clss = sess.run([self.res_scores, self.res_clss], feed_dict=feed_dict)
                bboxes = padded_rois.squeeze()
                bboxes = convert_bbox(bboxes, self.conv_feat_shape[:2], self.img_shape[:2])

            num_dets, scores, classes, bboxes = self.process_classifier_result(probs.squeeze(), clss.squeeze(), bboxes, test_dataset, k)
            
            det_scores.append(scores)
            det_classes.append(classes)
            det_bboxes.append(bboxes)
 
            img = cv2.imread(img_file)
            for i in range(num_dets):                               
                s = scores[i]
                y, x, h, w = bboxes[i]
                n = self.class_names[classes[i]]
                c = self.class_colors[classes[i]]
                cv2.rectangle(img, (x, y), (x+w-1, y+h-1), c, 2)
                cv2.putText(img, '%s : %.3f' %(n, s), (x+3, y+3), font, 0.5, (255, 255, 255), 1)

            if show_dets:
                winname = '%d detections' %num_dets
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 500, 100)
                cv2.waitKey()
                cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(result_dir, img_name+'_result.jpg'), img)

        results = {} 
        for i in range(test_dataset.count): 
            img_file = test_dataset.img_files[i]
            results[img_file] = []
            for s, c, b in zip(det_scores[i], det_classes[i], det_bboxes[i]): 
                results[img_file].append({'class_name': self.class_names[c], 'bbox':[b[1], b[0], b[3]-1, b[2]-1], 'score': s}) 

        pickle.dump(results, open(self.params.test_result_file, 'wb')) 
        print("Testing complete.") 

    def prepare_data_for_rpn(self, dataset, show_data=False):
        print("Preparing training data for RPN...")
        total_num_positive = 0
        total_num_negative = 0
        total_num_ambiguous = 0

        for i in tqdm(list(range(dataset.count))):
            img_file = dataset.img_files[i]
            H = dataset.img_heights[i]
            W = dataset.img_widths[i]           
            gt_classes = np.array(dataset.gt_classes[i]) 
            gt_bboxes = np.array(dataset.gt_bboxes[i])
            gt_bboxes = convert_bbox(gt_bboxes, [H, W], self.img_shape[:2])

            clss, bboxes, obj_clss= label_anchors(self.anchors, self.anchor_in_img, gt_classes, gt_bboxes) 
            regs = param_bbox(bboxes, self.anchors)
            anchor_file = dataset.anchor_files[i]
            np.savez(anchor_file, clss=clss, regs=regs, obj_clss=obj_clss)

            num_positive = len(np.where(clss==1)[0])
            total_num_positive += num_positive            

            num_negative = len(np.where(clss==0)[0])
            total_num_negative += num_negative            

            num_ambiguous = len(np.where(clss==-1)[0])
            total_num_ambiguous += num_ambiguous            

            if show_data:
                img = cv2.imread(img_file)
                targets = convert_bbox(bboxes, self.img_shape[:2], [H, W])
                scaled_anchors = convert_bbox(self.anchors, self.img_shape[:2], [H, W])
                for j in range(self.num_anchors):
                    y, x, h, w = targets[j]
                    cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (255, 0, 0), 2)
                    if clss[j]==1:
                        cv2.rectangle(img, (scaled_anchors[j][1], scaled_anchors[j][0]), (scaled_anchors[j][1]+scaled_anchors[j][3]-1, scaled_anchors[j][0]+scaled_anchors[j][2]-1), (255, 255, 255), 2)
                winname = '%d positve' %num_positive
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 100, 100)
                cv2.waitKey()
                cv2.destroyAllWindows()

        print("%d positive anchors, %d negative anchors, %d ambiguous anchors" %(total_num_positive, total_num_negative, total_num_ambiguous))

    def process_rpn_result(self, probs, rois):
        probs = probs[np.where(self.anchor_in_img==1)[0]]
        rois = rois[np.where(self.anchor_in_img==1)[0]]
        num_rois, _, top_k_rois = nms(probs, rois, self.num_rois)
        return num_rois, np.array(top_k_rois)

    def prepare_data_for_classifier(self, dataset, show_data=False):
        print("Preparing training data for Classifier...")
        font = cv2.FONT_HERSHEY_COMPLEX
        k = self.num_anchors_per_location
 
        total_num_rois = 0
        total_num_backgrounds = 0

        anchors = self.anchors.reshape((-1, k, 4))

        for i in tqdm(list(range(dataset.count))): 
            img_file = dataset.img_files[i]
            H = dataset.img_heights[i] 
            W = dataset.img_widths[i]

            anchor_file = dataset.anchor_files[i]
            anchor_data = np.load(anchor_file)

            anchor_clss = anchor_data['clss']
            anchor_obj_clss = anchor_data['obj_clss']
            anchor_regs = anchor_data['regs']

            num_objs = len(np.where(anchor_clss == 1)[0])
            ideal_num_backgrounds = max(min(self.num_rois - num_objs, num_objs), 6) * 1.0
            
            anchor_clss = anchor_clss.reshape((-1, k))            
            anchor_obj_clss = anchor_obj_clss.reshape((-1, k)) 
            anchor_regs = anchor_regs.reshape((-1, k, 4)) 

            num_rois = 0
            num_backgrounds = 0

            for j in range(k):
                current_anchors = anchors[:, j, :]

                current_anchor_clss = anchor_clss[:, j]
                current_anchor_regs = anchor_regs[:, j, :]
                current_anchor_obj_clss = anchor_obj_clss[:, j]

                positive_idx = np.where(current_anchor_clss == 1)[0] 
                negative_idx = np.where(current_anchor_clss == 0)[0] 

                num_positives = len(positive_idx)
                num_negatives = len(negative_idx)

                chosen_positive_idx = positive_idx
 
                if num_negatives > 0: 
                    ratio = ideal_num_backgrounds / (k * num_negatives) 
                    temp = np.random.uniform(0, 1, (num_negatives)) 
                    chosen_negative_idx = negative_idx[np.where(temp < ratio)[0]] 
                else:
                    chosen_negative_idx = np.array([], np.int32)

                chosen_idx = np.concatenate((chosen_positive_idx, chosen_negative_idx), axis=0)

                current_rois = current_anchors[chosen_idx] 
                current_regs = current_anchor_regs[chosen_idx] 

                positive_clss = current_anchor_obj_clss[chosen_positive_idx]
                negative_clss = np.array([self.num_classes-1] * (len(chosen_negative_idx)), np.int32)
                current_clss = np.concatenate((positive_clss, negative_clss), axis=0) 

                if j == 0:
                    rois = current_rois
                    clss = current_clss
                    regs = current_regs
                else:
                    rois = np.concatenate((rois, current_rois), axis=0)
                    clss = np.concatenate((clss, current_clss), axis=0)
                    regs = np.concatenate((regs, current_regs), axis=0)

            rois = rois[:self.num_rois]
            clss = clss[:self.num_rois]
            regs = regs[:self.num_rois]

            num_rois = len(clss)
            num_backgrounds = len(np.where(clss==self.num_classes-1)[0])

            roi_file = dataset.roi_files[i]
            rois1 = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            np.savez(roi_file, num=num_rois, rois=rois1, clss=clss, regs=regs)

            total_num_rois += num_rois
            total_num_backgrounds += num_backgrounds

            if show_data:
                img = cv2.imread(img_file)
                rois2 = convert_bbox(rois, self.img_shape[:2], [H, W]) 
                for j in range(len(clss)):
                    if clss[j]!=self.num_classes-1:                   
                        y, x, h, w = rois2[j]
                        c = self.class_colors[clss[j]]
                        cv2.rectangle(img, (x, y), (x+w-1, y+h-1), c, 2)                  
                winname = '%d rois: %d backgrounds' %(num_rois, num_backgrounds)
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 100, 100)
                cv2.waitKey() 
                cv2.destroyAllWindows()

        print(" %d RoIs are generated, in which %d are background" %(total_num_rois, total_num_backgrounds))

    def prepare_data_for_classifier2(self, dataset, show_data=False):
        print("Preparing training data for Classifier...")
        num_rois = self.num_rois
        font = cv2.FONT_HERSHEY_COMPLEX
 
        total_num_rois = 0
        total_num_backgrounds = 0
        for i in tqdm(list(range(dataset.count))): 
            img_file = dataset.img_files[i]
            H = dataset.img_heights[i] 
            W = dataset.img_widths[i]
            gt_classes = np.array(dataset.gt_classes[i]) 
            num_gts = gt_classes.shape[0]
            gt_bboxes = np.array(dataset.gt_bboxes[i])
            gt_bboxes = convert_bbox(gt_bboxes, [H, W], self.img_shape[:2])

            repetition = max(int(num_rois * 1.0 / num_gts), 1)
            tiled_gt_bboxes = np.tile(gt_bboxes, (repetition, 1))

            pertubation = np.random.uniform(-0.4, 0.4, tiled_gt_bboxes.shape)
            rois = unparam_bbox(pertubation, tiled_gt_bboxes, self.img_shape[:2])
            rois, clss, bboxes = label_rois(rois, gt_classes, gt_bboxes, self.num_classes-1)
            regs = param_bbox(bboxes, rois)

            roi_file = dataset.roi_files[i]
            scaled_rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            np.savez(roi_file, num=len(clss), rois=scaled_rois, clss=clss, regs=regs)

            total_num_rois += len(clss) 
            num_backgrounds = len(np.where(clss==self.num_classes-1)[0])
            total_num_backgrounds += num_backgrounds

            if show_data:
                img = cv2.imread(img_file)
                scaled_rois2 = convert_bbox(rois, self.img_shape[:2], [H, W]) 
                scaled_bboxes = convert_bbox(bboxes, self.img_shape[:2], [H, W])              
                for j in range(len(clss)):                 
                    y, x, h, w = scaled_rois2[j]
                    by, bx, bh, bw = scaled_bboxes[j]
                    c = self.class_colors[clss[j]]
                    cv2.rectangle(img, (x, y), (x+w-1, y+h-1), c, 2)
                    cv2.rectangle(img, (bx, by), (bx+bw-1, by+bh-1), (255, 0, 0), 2)
                winname = '%d rois: %d backgrounds' %(len(clss), num_backgrounds)
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 100, 100)
                cv2.waitKey() 
                cv2.destroyAllWindows()

        print("%d RoIs are generated, in which %d are background" %(total_num_rois, total_num_backgrounds))

    def process_classifier_result(self, probs, clss, bboxes, dataset, i):
        valid_idx = np.where(clss<self.num_classes-1)[0]
        dt_probs = probs[valid_idx]
        dt_clss = clss[valid_idx]
        dt_bboxes = bboxes[valid_idx]

        if len(valid_idx)==0:
            return 0, np.array([]), np.array([]), np.array([])

        num_dets, top_k_scores, top_k_classes, top_k_bboxes = postprocess(dt_probs, dt_clss, dt_bboxes, self.num_object_per_class)       

        H = dataset.img_heights[i]
        W = dataset.img_widths[i]
        top_k_bboxes = convert_bbox(top_k_bboxes, self.img_shape[:2], [H, W])

        return num_dets, np.array(top_k_scores), np.array(top_k_classes), np.array(top_k_bboxes)

    def save(self, sess):
        print("Saving model to %s" %self.save_dir)
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        print("Loading model...") 
        checkpoint = tf.train.get_checkpoint_state(self.save_dir) 
        if checkpoint is None: 
            print("Error: No saved model found. Please train first.") 
            sys.exit(0) 
        self.saver.restore(sess, checkpoint.model_checkpoint_path) 

    def load2(self, data_path, session, ignore_missing=True):
        print("Loading basic model from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                       #print("Variable %s:%s loaded" %(op_name, param_name))
                    except ValueError:
                        miss_count += 1
                       #print("Variable %s:%s missed" %(op_name, param_name))
                        if not ignore_missing:
                            raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))


