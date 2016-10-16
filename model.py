import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

from base_model import *
from utils.nn import *
from utils.bbox import *

class ObjectDetector(BaseModel):         

    def build(self):
        if self.basic_model=='vgg16':
            self.build_basic_vgg16()

        elif self.basic_model=='resnet50':
            self.build_basic_resnet50()

        elif self.basic_model=='resnet101':
            self.build_basic_resnet101()

        else:
            self.build_basic_resnet152()

        self.build_rpn()
        self.build_classifier()
        self.build_final()

    def build_basic_vgg16(self):
        print("Building the basic VGG16 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(self.img_loader.load_img(img_file))
        imgs = tf.pack(imgs)          

        conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
        conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
        conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
        conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
        conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
        conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
        conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
        conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
        conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

        self.conv_feats = conv5_3_feats
        self.conv_feat_shape = [40, 40, 512]

        self.roi_warped_feat_shape = [14, 14, 512]
        self.roi_pooled_feat_shape = [7, 7, 512]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic VGG16 net built.")

    def basic_block(self, input_feats, name1, name2, is_train, bn, c, s=2):
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, bn, c):
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_basic_resnet50(self):
        print("Building the basic ResNet50 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(self.img_loader.load_img(img_file))
        imgs = tf.pack(imgs)          

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, bn, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, bn, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, bn, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, bn, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, bn, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, bn, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, bn, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, bn, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, bn, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic ResNet50 net built.")

    def build_basic_resnet101(self):
        print("Building the basic ResNet101 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(self.img_loader.load_img(img_file))
        imgs = tf.pack(imgs)  

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic ResNet101 net built.")

    def build_basic_resnet152(self):
        print("Building the basic ResNet152 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(self.img_loader.load_img(img_file))
        imgs = tf.pack(imgs)  

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic ResNet152 net built.")
       
    def build_rpn(self):
        print("Building the RPN...")
        params = self.params
        bn = self.batch_norm
        is_train = self.is_train

        self.anchors, self.anchor_in_img, self.num_anchor_in_img = generate_anchors(self.img_shape[:2], self.conv_feat_shape[:2], self.anchor_scales, self.anchor_ratios)
        self.num_anchors = self.conv_feat_shape[0] * self.conv_feat_shape[1] * self.num_anchors_per_location

        feats = tf.placeholder(tf.float32, [self.batch_size]+self.conv_feat_shape) 
        gt_anchor_clss = tf.placeholder(tf.int32, [self.batch_size, self.num_anchors])
        gt_anchor_regs = tf.placeholder(tf.float32, [self.batch_size, self.num_anchors, 4])
        anchor_masks = tf.placeholder(tf.float32, [self.batch_size, self.num_anchors])

        self.feats = feats
        self.gt_anchor_clss = gt_anchor_clss
        self.gt_anchor_regs = gt_anchor_regs
        self.anchor_masks = anchor_masks

        rpn1 = convolution(feats, 5, 5, 512, 1, 1, 'rpn1', init_w='normal', stddev=0.01, group_id=1)
        rpn1 = nonlinear(rpn1, 'relu')

        rpn_logits = convolution(rpn1, 1, 1, 2*self.num_anchors_per_location, 1, 1, 'rpn_logits', init_w='normal',  stddev=0.01, group_id=1)
        rpn_logits = tf.reshape(rpn_logits, [-1, 2])

        if self.bbox_reg:
            rpn_regs = convolution(rpn1, 1, 1, 4*self.num_anchors_per_location, 1, 1, 'rpn_regs', init_w='normal', stddev=0.01, group_id=1)
            rpn_regs = tf.reshape(rpn_regs, [-1, 4])

        gt_anchor_clss = tf.reshape(gt_anchor_clss, [-1])       
        gt_anchor_regs = tf.reshape(gt_anchor_regs, [-1, 4])
        anchor_masks = tf.reshape(anchor_masks, [-1])

        loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_logits, gt_anchor_clss) * anchor_masks
        loss0 = tf.reduce_sum(loss0) / tf.reduce_sum(anchor_masks)

        if self.bbox_reg:
            anchor_reg_masks = anchor_masks * tf.to_float(gt_anchor_clss)
            w = self.smooth_l1_loss(rpn_regs, gt_anchor_regs) * anchor_reg_masks
            z = tf.reduce_sum(anchor_reg_masks)
            loss0 = tf.cond(tf.less(0.0, z), lambda: loss0 + params.rpn_reg_weight * tf.reduce_sum(w) / z, lambda: loss0)

        loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_1'))
        loss = loss0 + loss1

        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        opt_op = solver.minimize(loss, global_step=self.global_step)

        rpn_probs = tf.nn.softmax(rpn_logits)
        rpn_scores = tf.squeeze(tf.slice(rpn_probs, [0, 1], [-1, 1]))

        rpn_scores = tf.reshape(rpn_scores, [self.batch_size, self.num_anchors])

        self.rpn_loss = loss
        self.rpn_loss0 = loss0
        self.rpn_loss1 = loss1
        self.rpn_opt_op = opt_op

        self.rpn_scores = rpn_scores

        if self.bbox_reg:
            rpn_regs = tf.reshape(rpn_regs, [self.batch_size, self.num_anchors, 4])          
            self.rpn_regs = rpn_regs

        print("RPN built.")

    def smooth_l1_loss(self, s, t): # we compose known differentiable functions to implement smooth l1 loss function
        d = s - t
        x = 0.5 * d * d
        y = tf.nn.relu(d-1) + tf.nn.relu(-d-1)
        y = 0.5 * y * y
        loss = tf.reduce_sum(x-y, 1)
        return loss

    def build_classifier(self):
        print("Building the classifier...")
        params = self.params
        num_rois = self.num_rois
        is_train = self.is_train
        bn = self.batch_norm

        roi_warped_feats = tf.placeholder(tf.float32, [self.batch_size, num_rois]+self.roi_warped_feat_shape)  
        gt_roi_clss = tf.placeholder(tf.int32, [self.batch_size, num_rois]) 
        gt_roi_regs = tf.placeholder(tf.float32, [self.batch_size, num_rois, 4]) 
        roi_masks = tf.placeholder(tf.float32, [self.batch_size, num_rois]) 
        roi_reg_masks = tf.placeholder(tf.float32, [self.batch_size, num_rois]) 

        self.roi_warped_feats = roi_warped_feats
        self.gt_roi_clss = gt_roi_clss
        self.gt_roi_regs = gt_roi_regs
        self.roi_masks = roi_masks
        self.roi_reg_masks = roi_reg_masks
        
        roi_warped_feats = tf.reshape(roi_warped_feats, [self.batch_size*num_rois]+self.roi_warped_feat_shape)
        roi_pooled_feats = max_pool(roi_warped_feats, 2, 2, 2, 2, 'roi_pool')
        roi_pooled_feats = tf.reshape(roi_pooled_feats, [self.batch_size*num_rois, -1])

        fc6_feats = fully_connected(roi_pooled_feats, 4096, 'cls_fc6', init_w='normal', stddev=0.01, group_id=2)
        fc6_feats = nonlinear(fc6_feats, 'relu')
        fc6_feats = dropout(fc6_feats, 0.5, is_train)

        fc7_feats = fully_connected(fc6_feats, 4096, 'cls_fc7', init_w='normal', stddev=0.01, group_id=2)
        fc7_feats = nonlinear(fc7_feats, 'relu')
        fc7_feats = dropout(fc7_feats, 0.5, is_train)

        logits = fully_connected(fc7_feats, self.num_classes, 'cls_logits', init_w='normal', stddev=0.01, group_id=2)

        gt_roi_clss = tf.reshape(gt_roi_clss, [-1])
        gt_roi_regs = tf.reshape(gt_roi_regs, [-1, 4])
        roi_masks = tf.reshape(roi_masks, [-1])
        roi_reg_masks = tf.reshape(roi_reg_masks, [-1])

        if self.bbox_reg:
            if self.bbox_per_class:
                regs = fully_connected(fc7_feats, 4*self.num_classes, 'cls_reg', init_w='normal', stddev=0.001, group_id=2)
                relevant_regs = []
                for i in range(self.batch_size*num_rois):
                    relevant_regs.append(tf.squeeze(tf.slice(regs, [i, 4*gt_roi_clss[i]], [1, 4])))
                relevant_regs = tf.pack(relevant_regs) 
            else:
                regs = fully_connected(fc7_feats, 4, 'cls_reg', init_w='normal', stddev=0.001, group_id=2)
                relevant_regs = regs

        loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, gt_roi_clss) * roi_masks
        loss0 = tf.reduce_sum(loss0) / tf.reduce_sum(roi_masks)
    
        if self.bbox_reg:
            w = self.smooth_l1_loss(relevant_regs, gt_roi_regs) * roi_reg_masks
            z = tf.reduce_sum(roi_reg_masks)
            loss0 = tf.cond(tf.less(0.0, z), lambda: loss0 + params.cls_reg_weight * tf.reduce_sum(w) / z, lambda: loss0)

        loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_2'))
        loss = loss0 + loss1

        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        opt_op = solver.minimize(loss, global_step=self.global_step)

        probs = tf.nn.softmax(logits)
        clss = tf.argmax(probs, 1)
        scores = tf.reduce_max(probs, 1) 
        scores = scores * roi_masks

        res_clss = tf.reshape(clss, [self.batch_size, num_rois])
        res_scores = tf.reshape(scores, [self.batch_size, num_rois])

        self.cls_loss = loss
        self.cls_loss0 = loss0
        self.cls_loss1 = loss1
        self.cls_opt_op = opt_op

        self.res_clss = res_clss
        self.res_scores = res_scores

        if self.bbox_reg:
            if self.bbox_per_class:
                res_regs = []
                for i in range(self.batch_size*num_rois):
                    res_regs.append(tf.squeeze(tf.slice(regs, [i, 4*clss[i]], [1, 4])))
                res_regs = tf.pack(res_regs) 
            else:
                res_regs = regs
            res_regs = tf.reshape(res_regs, [self.batch_size, num_rois, 4])
            self.res_regs = res_regs
  
        print("Classifier built.")

    def build_final(self):
        params = self.params

        loss0 = params.rpn_weight * self.rpn_loss0 + params.cls_weight * self.cls_loss0
        loss1 = params.weight_decay * (tf.add_n(tf.get_collection('l2_1')) + tf.add_n(tf.get_collection('l2_2')))
        loss = loss0 + loss1

        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        opt_op = solver.minimize(loss, global_step=self.global_step)

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op

    def get_roi_feats(self, feats, rois):
        roi_warped_feats = []
        for i in range(self.batch_size):
            current_feats = feats[i]
            current_rois = rois[i]
            roi_warped_feats.append(self.roi_warp(current_feats, current_rois))
        roi_warped_feats = np.array(roi_warped_feats)
        return roi_warped_feats

    def roi_warp(self, feats, rois):  
        ch, cw, c = self.conv_feat_shape
        th, tw, c = self.roi_warped_feat_shape
        num_rois = self.num_rois
        warped_feats = []

        for k in range(num_rois):
            y, x, h, w = rois[k, 0], rois[k, 1], rois[k, 2], rois[k, 3] 

            j = np.array(list(range(h)), np.float32)
            i = np.array(list(range(w)), np.float32)
            tj = np.array(list(range(th)), np.float32)
            ti = np.array(list(range(tw)), np.float32)

            j = np.expand_dims(np.expand_dims(np.expand_dims(j, 1), 2), 3)
            i = np.expand_dims(np.expand_dims(np.expand_dims(i, 0), 2), 3)
            tj = np.expand_dims(np.expand_dims(np.expand_dims(tj, 1), 0), 1)
            ti = np.expand_dims(np.expand_dims(np.expand_dims(ti, 0), 0), 1)

            j = np.tile(j, (1, w, th, tw)) 
            i = np.tile(i, (h, 1, th, tw)) 
            tj = np.tile(tj, (h, w, 1, tw)) 
            ti = np.tile(ti, (h, w, th, 1)) 

            b = tj * h * 1.0 / th - j
            a = ti * w * 1.0 / tw - i

            b = np.maximum(np.zeros_like(b), 1 - np.absolute(b))
            a = np.maximum(np.zeros_like(a), 1 - np.absolute(a))

            G = b * a
            G = G.reshape((h*w, th*tw))

            sliced_feat = feats[y:y+h, x:x+w, :]
            sliced_feat = sliced_feat.swapaxes(0, 1)
            sliced_feat = sliced_feat.swapaxes(0, 2)
            sliced_feat = sliced_feat.reshape((-1, h*w))

            warped_feat = np.matmul(sliced_feat, G)
            warped_feat = warped_feat.reshape((-1, th, tw))
            warped_feat = warped_feat.swapaxes(0, 1)
            warped_feat = warped_feat.swapaxes(1, 2)

            warped_feats.append(warped_feat)

        warped_feats = np.array(warped_feats)
        return warped_feats

    def get_feed_dict_for_rpn(self, batch, is_train, feats):
        if is_train:
            img_files, anchor_files = batch
            gt_anchor_clss, gt_anchor_regs, anchor_masks = self.process_anchor_data(anchor_files)                 
            return {self.feats: feats, self.gt_anchor_clss: gt_anchor_clss, self.gt_anchor_regs: gt_anchor_regs, self.anchor_masks: anchor_masks, self.is_train: is_train}

        else:
            return {self.feats: feats, self.is_train: is_train}

    def get_feed_dict_for_classifier(self, batch, is_train, feats, rois=None, masks=None):
        if is_train:
            _, roi_files = batch
            rois, gt_roi_clss, gt_roi_regs, roi_masks, roi_reg_masks = self.process_roi_data(roi_files)
            roi_warped_feats = self.get_roi_feats(feats, rois)
            return {self.roi_warped_feats: roi_warped_feats, self.gt_roi_clss: gt_roi_clss, self.gt_roi_regs: gt_roi_regs, self.roi_masks: roi_masks, self.roi_reg_masks: roi_reg_masks, self.is_train: is_train}

        else:
            roi_warped_feats = self.get_roi_feats(feats, rois)
            return {self.roi_warped_feats: roi_warped_feats, self.roi_masks: masks, self.is_train: is_train}

    def get_feed_dict_for_all(self, batch, is_train, feats=None):
        if is_train:
            _, anchor_files, roi_files = batch
            gt_anchor_clss, gt_anchor_regs, anchor_masks = self.process_anchor_data(anchor_files)
            rois, gt_roi_clss, gt_roi_regs, roi_masks, roi_reg_masks = self.process_roi_data(roi_files)
            roi_warped_feats = self.get_roi_feats(feats, rois)
            return {self.feats: feats, self.gt_anchor_clss: gt_anchor_clss, self.gt_anchor_regs: gt_anchor_regs, self.anchor_masks: anchor_masks, self.roi_warped_feats: roi_warped_feats, self.gt_roi_clss: gt_roi_clss, self.gt_roi_regs: gt_roi_regs, self.roi_masks: roi_masks, self.roi_reg_masks: roi_reg_masks, self.is_train: is_train}
            
        else:  
            img_files = batch  
            return {self.img_files: img_files, self.is_train: is_train}

    def process_anchor_data(self, anchor_files):
        gt_anchor_clss = []
        gt_anchor_regs = []
        anchor_masks = []

        for i in range(self.batch_size):
            anchor_data = np.load(anchor_files[i])
            clss = anchor_data['clss']
            regs = anchor_data['regs']

            masks = sample_anchors(clss, self.num_anchors_per_location)
            clss[np.where(clss==-1)[0]] = 0

            gt_anchor_clss.append(clss)
            gt_anchor_regs.append(regs)
            anchor_masks.append(masks)

        gt_anchor_clss = np.array(gt_anchor_clss)
        gt_anchor_regs = np.array(gt_anchor_regs)
        anchor_masks = np.array(anchor_masks)

        return gt_anchor_clss, gt_anchor_regs, anchor_masks

    def process_roi_data(self, roi_files):
        num_rois = self.num_rois
        rois = []
        gt_roi_clss = []
        gt_roi_regs = []
        roi_masks = []
        roi_reg_masks = []

        for i in range(self.batch_size):
            roi_data = np.load(roi_files[i])
            num_real_rois = roi_data['num']

            current_rois = np.ones((num_rois, 4), np.int32) * 3
            current_rois[:num_real_rois] = roi_data['rois']

            current_roi_clss = np.ones((num_rois), np.int32)
            current_roi_clss[:num_real_rois] = roi_data['clss']

            current_roi_regs = np.ones((num_rois, 4), np.float32)
            current_roi_regs[:num_real_rois] = roi_data['regs']
                                       
            current_roi_masks = np.zeros((num_rois), np.float32)
            current_roi_masks[:num_real_rois] = 1.0
                       
            current_roi_reg_masks = current_roi_masks * 1.0
            current_roi_reg_masks[np.where(current_roi_clss==self.num_classes-1)[0]] = 0.0

            rois.append(current_rois)
            gt_roi_clss.append(current_roi_clss)
            gt_roi_regs.append(current_roi_regs)
            roi_masks.append(current_roi_masks)
            roi_reg_masks.append(current_roi_reg_masks)

        rois = np.array(rois)
        gt_roi_clss = np.array(gt_roi_clss)
        gt_roi_regs = np.array(gt_roi_regs)
        roi_masks = np.array(roi_masks)
        roi_reg_masks = np.array(roi_reg_masks)

        return rois, gt_roi_clss, gt_roi_regs, roi_masks, roi_reg_masks

