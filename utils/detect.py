#### 需要进行转tensor再进行预测


import numpy as np
import paddle

from paddle.vision.transforms import Compose

from utils.utils import Rescale, Normailize, Reshape
from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes


def decoder(ims, anchors, cls_score, bbox_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):  # 验证阶段一个一个来的
    box_coder = BoxCoder()
    if test_conf is not None:
        thresh = test_conf
    bboxes = box_coder.decode(anchors, bbox_pred, mode='xywht')  # 原始锚框与预测出来的偏移量进行解码，还原原始5个点
    bboxes = clip_boxes(bboxes, ims)
    scores = paddle.max(cls_score, axis=2, keepdim=True)       # [[[0.9612],[0.7785],[0.9859],...,]]] Paddle  [2,40029,1] 
    keep = (scores >= thresh)[0, :, 0]  # [2,40029,1] -----> [400029 True False]
    nms_scores, nms_class, output_boxes = [paddle.zeros([1]), paddle.zeros([1]), paddle.zeros([1, 5])]
    if keep.sum() == 0:
        return nms_scores, nms_class, output_boxes
    scores = paddle.unsqueeze(scores[0][keep], axis=0)
    anchors = paddle.unsqueeze(anchors[0][keep], axis=0)
    cls_score = paddle.unsqueeze(cls_score[0][keep], axis=0)
    bboxes = paddle.unsqueeze(bboxes[0][keep], axis=0)
    anchors_nms_idx = nms(np.array(paddle.concat(x=[bboxes, scores], axis=2)[0]), nms_thresh)

    temp = paddle.to_tensor(cls_score[0][anchors_nms_idx])
    nms_scores = paddle.max(temp, axis=1)
    nms_class = paddle.argmax(temp, axis=1)
    output_boxes = paddle.concat([
        paddle.to_tensor(bboxes[0][anchors_nms_idx]),
        paddle.to_tensor(anchors[0][anchors_nms_idx])],
        axis=1
    )
    return [nms_scores, nms_class, output_boxes]


def im_detect(model, src, target_sizes, use_gpu=True, conf=None):
    if isinstance(target_sizes, int):
        target_sizes = [target_sizes]
    if len(target_sizes) == 1:
        return single_scale_detect(model, src, target_size=target_sizes[0], use_gpu=use_gpu, conf=conf)
    else:
        ms_dets = None
        for ind, scale in enumerate(target_sizes):
            cls_dets = single_scale_detect(model, src, target_size=scale, use_gpu=use_gpu, conf=conf)
            if cls_dets.shape[0] == 0:
                continue
            if ms_dets is None:
                ms_dets = cls_dets
            else:
                ms_dets = np.vstack((ms_dets, cls_dets))
        if ms_dets is None:
            return np.zeros((0, 7))
        cls_dets = np.hstack((ms_dets[:, 2:7], ms_dets[:, 1][:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, 0.1)
        return ms_dets[keep, :]


def single_scale_detect(model, src, target_size, use_gpu=True, conf=None):
    im, im_scales = Rescale(target_size=target_size, keep_ratio=True)(src)
    im = Compose([Normailize(), Reshape(unsqueeze=True)])(im)
    # if use_gpu:
    #     model, im = model.cuda(), im.cuda()  ############改过##############
    with paddle.no_grad():   ############ 大改
        ims, anchors, cls_score, bbox_pred, test_conf = model(im, test_conf=conf)
        # scores, classes, boxes = model(im, test_conf=conf)
        scores, classes, boxes = decoder(ims, anchors, cls_score, bbox_pred, test_conf=test_conf)
    scores = scores.cpu().numpy()  ############改过##############
    classes = classes.cpu().numpy()  ############改过##############
    boxes = boxes.cpu().numpy()  ############改过##############
    boxes[:, :4] = boxes[:, :4] / im_scales
    if boxes.shape[1] > 5:
        boxes[:, 5:9] = boxes[:, 5:9] / im_scales
    scores = np.reshape(scores, (-1, 1))
    classes = np.reshape(classes, (-1, 1))
    cls_dets = np.concatenate([classes, scores, boxes], axis=1)
    keep = np.where(classes > 0)[0]
    return cls_dets[keep, :]
    # cls, score, x,y,x,y,a,   a_x,a_y,a_x,a_y,a_a
