import time
import numpy as np

def iou_bbox(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1, np.float32)
    bboxes2 = np.array(bboxes2, np.float32)
    
    overlap_min_y = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    overlap_max_y = np.minimum(bboxes1[:, 0] + bboxes1[:, 2] - 1, bboxes2[:, 0] + bboxes2[:, 2] - 1)
    overlap_height = np.maximum(overlap_max_y - overlap_min_y + 1, np.zeros_like(bboxes1[:, 0]))

    overlap_min_x = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    overlap_max_x = np.minimum(bboxes1[:, 1] + bboxes1[:, 3] - 1, bboxes2[:, 1] + bboxes2[:, 3] - 1)
    overlap_width = np.maximum(overlap_max_x - overlap_min_x + 1, np.zeros_like(bboxes1[:, 1]))

    area_overlap = overlap_height * overlap_width
    area_union = bboxes1[:, 2] * bboxes1[:, 3] + bboxes2[:, 2] * bboxes2[:, 3] - area_overlap
    
    iou = area_overlap / area_union
    return iou

def param_bbox(bboxes, anchors):
    bboxes = np.array(bboxes, np.float32)
    anchors = np.array(anchors, np.float32)

    tyx = (bboxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    thw = np.log(bboxes[:, 2:] / anchors[:, 2:])

    t = np.concatenate((tyx, thw), axis=1)
    return t

def unparam_bbox(t, anchors, max_shape=None):
    t = np.array(t, np.float32)
    anchors = np.array(anchors, np.float32)

    yx = t[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = np.exp(t[:, 2:]) * anchors[:, 2:]

    bboxes = np.concatenate((yx, hw), axis=1)

    if max_shape != None:
        bboxes = rectify_bbox(bboxes, max_shape)

    return bboxes

def rectify_bbox(bboxes, max_shape):
    bboxes = np.array(bboxes, np.int32)
    n = bboxes.shape[0]
    if n == 0:
        return bboxes

    h, w = max_shape

    bboxes[:, 0] = np.maximum(bboxes[:, 0], np.zeros((n)))
    bboxes[:, 0] = np.minimum(bboxes[:, 0], (h-1) * np.ones((n)))
    bboxes[:, 1] = np.maximum(bboxes[:, 1], np.zeros((n)))
    bboxes[:, 1] = np.minimum(bboxes[:, 1], (w-1) * np.ones((n)))
    bboxes[:, 2] = np.maximum(bboxes[:, 2], np.ones((n)))
    bboxes[:, 2] = np.minimum(bboxes[:, 2], h * np.ones((n)) - bboxes[:, 0])
    bboxes[:, 3] = np.maximum(bboxes[:, 3], np.ones((n)))
    bboxes[:, 3] = np.minimum(bboxes[:, 3], w * np.ones((n)) - bboxes[:, 1])

    return bboxes
    
def convert_bbox(bboxes, old_shape, new_shape):
    bboxes = np.array(bboxes, np.float32)
    if bboxes.shape[0] == 0:
        return bboxes

    oh, ow = old_shape
    nh, nw = new_shape

    bboxes[:, 0] = bboxes[:, 0] * nh / oh
    bboxes[:, 1] = bboxes[:, 1] * nw / ow
    bboxes[:, 2] = bboxes[:, 2] * nh / oh
    bboxes[:, 3] = bboxes[:, 3] * nw / ow

    bboxes = rectify_bbox(bboxes, new_shape)
    return bboxes

def generate_anchors(img_shape, feat_shape, scales, ratios):
    ih, iw = img_shape
    fh, fw = feat_shape
    ls = len(scales)
    lr = len(ratios)
    n = fh * fw * ls * lr
   
    j = np.array(list(range(fh)))
    j = np.expand_dims(np.expand_dims(np.expand_dims(j, 1), 2), 3)
    j = np.tile(j, (1, fw, ls, lr))
    j = j.reshape((-1))

    i = np.array(list(range(fw)))
    i = np.expand_dims(np.expand_dims(np.expand_dims(i, 0), 2), 3)
    i = np.tile(i, (fh, 1, ls, lr))
    i = i.reshape((-1))

    s = np.array(scales)
    s = np.expand_dims(np.expand_dims(np.expand_dims(s, 0), 1), 3)
    s = np.tile(s, (fh, fw, 1, lr))
    s = s.reshape((-1))

    r = np.array(ratios)
    r = np.expand_dims(np.expand_dims(np.expand_dims(r, 0), 1), 2)
    r = np.tile(r, (fh, fw, ls, 1, 1))
    r = r.reshape((-1, 2))
  
    y = (j + 0.5) * ih / fh - s * r[:, 0] / 2.0
    x = (i + 0.5) * iw / fw - s * r[:, 1] / 2.0
    h = s * r[:, 0]
    w = s * r[:, 1]

    anchor_in_img = np.ones((n), np.int32)  
    anchor_in_img[np.where(y<0)[0]] = 0
    anchor_in_img[np.where(x<0)[0]] = 0
    anchor_in_img[np.where(h+y>ih)[0]] = 0
    anchor_in_img[np.where(w+x>iw)[0]] = 0

    y = np.maximum(y, np.zeros((n)))
    x = np.maximum(x, np.zeros((n)))
    h = np.minimum(h, ih-y)
    w = np.minimum(w, iw-x)

    y = np.expand_dims(y, 1)
    x = np.expand_dims(x, 1)
    h = np.expand_dims(h, 1)
    w = np.expand_dims(w, 1)
    anchors = np.concatenate((y, x, h, w), axis=1)

    anchor_in_img1 = anchor_in_img.reshape((fh*fw, ls*lr))
    num_anchor_in_img = np.sum(anchor_in_img1, axis=0)

    return anchors, anchor_in_img, num_anchor_in_img

def label_anchors(anchors, anchor_in_img, gt_classes, gt_bboxes, iou_low_threshold=0.4, iou_high_threshold=0.6):
    n = anchors.shape[0]
    k = gt_bboxes.shape[0]
    
    tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
    tiled_gt_bboxes = np.tile(np.expand_dims(gt_bboxes, 0), (n, 1, 1))

    tiled_anchors = tiled_anchors.reshape((-1, 4))
    tiled_gt_bboxes = tiled_gt_bboxes.reshape((-1, 4))

    ious = iou_bbox(tiled_anchors, tiled_gt_bboxes)
    ious = ious.reshape(n, k)

    max_iou = np.amax(ious, axis=1)

    best_gt_bbox_ids = np.argmax(ious, axis=1)
    bboxes = gt_bboxes[best_gt_bbox_ids]
    classes = gt_classes[best_gt_bbox_ids]

 #   best_anchor_ids = np.argmax(ious, axis=0)

    labels = -np.ones((n), np.int32)

    positive_idx = np.where(max_iou >= iou_high_threshold)[0]
    labels[positive_idx] = 1

    negative_idx = np.where(max_iou < iou_low_threshold)[0]
    labels[negative_idx] = 0

    ignore_idx = np.where(anchor_in_img == 0)[0]
    labels[ignore_idx] = -1

 #   labels[best_anchor_ids] = 1

    return labels, bboxes, classes

def sample_anchors(labels, k):
    l = len(labels)
    n = l/k
    labels = labels.reshape((n, k))

    masks = np.zeros((n, k), np.float32)

    for i in range(k):
        current_labels = labels[:, i] 

        positive_idx = np.where(current_labels == 1)[0] 
        negative_idx = np.where(current_labels == 0)[0] 

        num_positives = len(positive_idx)
        num_negatives = len(negative_idx)

        current_masks = np.zeros((n), np.float32) 

        if num_positives + num_negatives <= 12: 
            current_masks[positive_idx] = 1.0 
            current_masks[negative_idx] = 1.0 

        else:
            if num_positives > 0: 
                positive_ratio = 4.0 / num_positives 
                temp = np.random.uniform(0, 1, (num_positives)) 
                current_masks[positive_idx] = (temp < positive_ratio) * 1.0 

            if num_negatives > 0: 
                negative_ratio = 8.0 / num_negatives 
                temp = np.random.uniform(0, 1, (num_negatives)) 
                current_masks[negative_idx] = (temp < negative_ratio) * 1.0 

        masks[:, i] = current_masks 

    masks = masks.reshape((-1))

    return masks 

def label_rois(rois, gt_classes, gt_bboxes, background_id, iou_low_thresh1old=0.4, iou_high_threshold=0.6):
    n = rois.shape[0]
    k = len(gt_classes)
    
    tiled_rois = np.tile(np.expand_dims(rois, 1), (1, k, 1))
    tiled_gt_bboxes = np.tile(np.expand_dims(gt_bboxes, 0), (n, 1, 1))
    
    tiled_rois = tiled_rois.reshape((-1, 4))
    tiled_gt_bboxes = tiled_gt_bboxes.reshape((-1, 4))
    
    ious = iou_bbox(tiled_rois, tiled_gt_bboxes)
    ious = ious.reshape(n, k)
    
    max_iou = np.amax(ious, axis=1)

    best_gt_bbox_ids = np.argmax(ious, axis=1)
    best_gt_bboxes = gt_bboxes[best_gt_bbox_ids]

    positive_idxs = np.where(max_iou>=iou_high_threshold)[0]
    negative_idxs = np.where(max_iou<iou_low_threshold)[0]
    chosen_idxs = np.concatenate((positive_idxs, negative_idxs), axis=0)

    rois = rois[chosen_idxs]
    bboxes = best_gt_bboxes[chosen_idxs]

    positive_labels = gt_classes[best_gt_bbox_ids[positive_idxs]]        
    negative_labels = np.ones([background_id] * len(negative_idxs), np.int32)
    classes = np.concatenate((positive_labels, negative_labels), axis=0)

    return rois, classes, bboxes

def nms(scores, bboxes, k, iou_threshold=0.7):
    n = len(scores)

    idx = np.argsort(scores)[::-1]
    sorted_scores = scores[idx]
    sorted_bboxes = bboxes[idx]
    
    top_k_ids = []
    size = 0
    i = 0

    while i < n and size < k:
        if sorted_scores[i] < 0.5:
            break
        top_k_ids.append(i)
        size += 1
        i += 1
        while i < n:
            tiled_bbox_i = np.tile(sorted_bboxes[i], (size, 1)) 
            ious = iou_bbox(tiled_bbox_i, sorted_bboxes[top_k_ids])
            if np.amax(ious) > iou_threshold:
                i += 1
            else:
                break

    return size, sorted_scores[top_k_ids], sorted_bboxes[top_k_ids]

def postprocess(scores, classes, bboxes, k, iou_threshold=0.3):
    n = len(scores)
 
    count_per_cls = {cls:0 for cls in classes}
    bbox_per_cls = {cls:[] for cls in classes}
    score_per_cls = {cls:[] for cls in classes}

    for i in range(n):
        count_per_cls[classes[i]] += 1
        bbox_per_cls[classes[i]] += [bboxes[i]]
        score_per_cls[classes[i]] += [scores[i]]
        
    dt_num = 0
    dt_classes = []    
    dt_scores = []
    dt_bboxes = []

    for cls in count_per_cls:
        current_count = count_per_cls[cls]
        current_scores = np.array(score_per_cls[cls], np.float32)
        current_bboxes = np.array(bbox_per_cls[cls], np.int32)

        idx = np.argsort(current_scores)[::-1]
        max_score = current_scores[idx[0]]
        sorted_scores = current_scores[idx]
        sorted_bboxes = current_bboxes[idx]

        top_k_ids = []
        size = 0
        i = 0

        while i < current_count and size < k:
            if sorted_scores[i] < max(0.75 * max_score, 0.5):
                break
            top_k_ids.append(i)
            dt_num += 1
            dt_classes.append(cls)
            dt_scores.append(sorted_scores[i])
            dt_bboxes.append(sorted_bboxes[i])
            size += 1
            i += 1

            while i < current_count:
                tiled_bbox_i = np.tile(sorted_bboxes[i], (size, 1))
                ious = iou_bbox(tiled_bbox_i, sorted_bboxes[top_k_ids])
                if np.amax(ious) > iou_threshold:
                    i += 1
                else:
                    break

    return dt_num, np.array(dt_scores, np.float32), np.array(dt_classes, np.int32), np.array(dt_bboxes, np.int32)

