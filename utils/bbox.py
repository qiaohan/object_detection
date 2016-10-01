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
    h, w = max_shape
    n, _ = bboxes.shape
    bboxes = np.array(bboxes, np.int32)

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
    oh, ow = old_shape
    nh, nw = new_shape
    bboxes = np.array(bboxes, np.float32)

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

    return anchors, anchor_in_img


def label_anchors(anchors, anchor_in_img, gt_bboxes, iou_low_threshold=0.3, iou_high_threshold=0.7):
    n, _ = anchors.shape
    k, _ = gt_bboxes.shape
    
    tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
    tiled_gt_bboxes = np.tile(np.expand_dims(gt_bboxes, 0), (n, 1, 1))

    tiled_anchors = tiled_anchors.reshape((-1, 4))
    tiled_gt_bboxes = tiled_gt_bboxes.reshape((-1, 4))

    tiled_anchor_in_img = np.tile(np.expand_dims(anchor_in_img, 1), (1, k))
    tiled_anchor_in_img = tiled_anchor_in_img.reshape((-1)) 

    ious = iou_bbox(tiled_anchors, tiled_gt_bboxes)
    ious = ious * tiled_anchor_in_img   # ignore the anchors which corss the boundaries during training
    ious = ious.reshape(n, k)

    max_iou = np.amax(ious, axis=1)
 
    best_gt_bbox_ids = np.argmax(ious, axis=1)
    best_gt_bboxes = gt_bboxes[best_gt_bbox_ids]

    best_anchor_ids = np.argmax(ious, axis=0)

    labels = -np.ones((n), np.int32)
    labels[np.where(max_iou >= iou_high_threshold)[0]] = 1
    labels[np.where(max_iou < iou_low_threshold)[0]] = 0
    labels[best_anchor_ids] = 1

    return labels, best_gt_bboxes


def sample_anchors(labels):
    P = (labels == 1) * 1.0
    N = (labels == 0) * 1.0

    ratio = min(np.sum(P) * 1.5 / np.sum(N), 1.0)
    temp = np.random.uniform(0, 1, (len(labels)))
    masks = P + N * (temp < ratio)

    return masks.astype(np.float32)


def label_rois(rois, gt_classes, gt_bboxes, background_id, iou_threshold=0.5):
    n, _ = rois.shape
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

    labels = gt_classes[best_gt_bbox_ids]
    labels[np.where(max_iou < iou_threshold)[0]] = background_id

    return labels, best_gt_bboxes


def nms(scores, bboxes, k, iou_threshold=0.7):
    n = len(scores)

    idx = np.argsort(scores)[::-1]
    sorted_scores = scores[idx]
    sorted_bboxes = bboxes[idx]
    
    top_k_ids = []
    size = 0
    i = 0

    while i < n and size < k:
        if sorted_scores[i] < 0.6:
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


def nms2(scores, classes, bboxes, k, iou_threshold=0.5):
    n = len(scores)

    idx = np.argsort(scores)[::-1]
    sorted_scores = scores[idx]
    sorted_classes = classes[idx]
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

    return size, sorted_scores[top_k_ids], sorted_classes[top_k_ids], sorted_bboxes[top_k_ids]

