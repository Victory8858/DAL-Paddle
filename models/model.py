# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('/home/aistudio/work/DAL')
sys.path.append('/home/aistudio/work/DAL/datasets/DOTA_devkit')
import paddle
import paddle.vision.models as models
from models.anchors import Anchors
from models.fpn import FPN, LastLevelP6P7
from models.heads import CLSHead, REGHead  # , MultiHead
from models.losses import IntegratedLoss  # , KLLoss

from datasets import *
from utils.utils import *
from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes, rbox_2_quad
# from ppdet.modeling.ops import multiclass_nms

class RetinaNet(paddle.nn.Layer):
    def __init__(self, backbone='res50', hyps=None):
        super(RetinaNet, self).__init__()
        self.num_classes = 16
        self.anchor_generator = Anchors(ratios=np.array([0.5, 1, 2]))
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)

        self.fpn = FPN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256),
            use_asff=False
        )
        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )
        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5  # xywha
        )
        self.loss = IntegratedLoss(func='smooth')
        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        self.backbone = models.resnet50(pretrained=True)
        self.fpn_in_channels = [512, 1024, 2048]
        del self.backbone.avgpool
        del self.backbone.fc

    def ims_2_features(self, ims):
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(ims)))
        c2 = self.backbone.layer1(self.backbone.maxpool(c1))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        # c_i shape: bs,C,H,W
        return [c3, c4, c5]

    def forward(self, ims, test_conf=None, gt_boxes=None, process=None):
        anchors_list, offsets_list, cls_list, var_list = [], [], [], []
        original_anchors = self.anchor_generator(ims)  # (batch_size, num_all_achors, 5)
        anchors_list.append(original_anchors)  # [2, 40029, 5] 原始锚框
        # print('anchors_list:', anchors_list)
        features = self.fpn(self.ims_2_features(ims))  # 生成特征图[2, 256, 100, 100] [2, 256, 50, 50] [2, 256, 25, 25] [2, 256, 13, 13] [2, 256, 7, 7]

        # 原始版本
        # cls_score = paddle.concat([self.cls_head(feature) for feature in features], axis=1)  # [2, 40029, 16]  0.01000000, 0.01000000, 0.01000000, ..., 0.01000000
        # bbox_pred = paddle.concat([self.reg_head(feature) for feature in features], axis=1)  # [2, 40029, 5] [-0.02714705,  0.49628371, -0.28771776, -0.04605314, -0.83333385]加了初始化全0
        
        # 模型动转静出现问题
        cls_score_list = []
        bbox_pred_list = []
        for feature in features:
            cls_score_list.append(self.cls_head(feature))
            bbox_pred_list.append(self.reg_head(feature))
        cls_score = paddle.concat(cls_score_list, axis=1)
        bbox_pred = paddle.concat(bbox_pred_list, axis=1)
        
        bboxes = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='xywht').detach()

        # return bbox_pred, bbox_pred
        if self.training:
            losses = dict()
            bf_weight = self.calc_mining_param(process, 0.3)
            losses['loss_cls'], losses['loss_reg'] = self.loss(cls_score, bbox_pred, anchors_list[-1], bboxes, gt_boxes, \
                                                               md_thres=0.6, mining_param=(bf_weight, 1 - bf_weight, 5))
            return losses

        else:  # eval() mode
            return ims, anchors_list[-1], cls_score, bbox_pred, test_conf
            # return self.decoder(ims, anchors_list[-1], cls_score, bbox_pred, test_conf=test_conf)  # 变回原始

    # def decoder(self, ims, anchors, cls_score, bbox_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):  # 验证阶段一个一个来的
    #     if test_conf is not None:
    #         thresh = test_conf
    #     bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')  # 原始锚框与预测出来的偏移量进行解码，还原原始5个点
    #     bboxes = clip_boxes(bboxes, ims)
    #     # print('anchors:', anchors)  # [1, 40029, 5] 生成的原始锚框
    #     # print('bboxes:', bboxes)
    #     # print('bboxes:', bboxes)
    #     # print('bboxes shape:', bboxes.shape)
    #     # scores = paddle.max(cls_score, axis=2, keepdim=True)[0]  # [[[0.9612],[0.7785],[0.9859],...,]]] Pytorch的max返回两个，一个max，一个argmax
    #     # print('cls score:', cls_score)
    #     # print('cls score shape:', cls_score.shape)
    #     scores = paddle.max(cls_score, axis=2, keepdim=True)       # [[[0.9612],[0.7785],[0.9859],...,]]] Paddle  [2,40029,1] 
    #     # print('scores:', scores)
    #     # print('scores shape:', scores.shape)
    #     keep = (scores >= thresh)[0, :, 0]  # [2,40029,1] -----> [400029 True False]
    #     nms_scores, nms_class, output_boxes = [paddle.zeros([1]), paddle.zeros([1]), paddle.zeros([1, 5])]
    #     if keep.sum() == 0:
    #         return nms_scores, nms_class, output_boxes
    #     scores = paddle.unsqueeze(scores[0][keep], axis=0)
    #     anchors = paddle.unsqueeze(anchors[0][keep], axis=0)
    #     # cls_score_nms = paddle.transpose(cls_score, perm=[0,2,1]) ##############
    #     cls_score = paddle.unsqueeze(cls_score[0][keep], axis=0)
    #     bboxes = paddle.unsqueeze(bboxes[0][keep], axis=0)
    #     # print('bboxes:', bboxes)
    #     # bboxes = rbox_2_quad(bboxes[0], mode='xywha')
    #     # print('bboxes:', bboxes)

    #     # out, anchors_nms_idx = multiclass_nms(bboxes=bboxes, scores=cls_score_nms, score_threshold=0.5, nms_top_k=400, nms_threshold=nms_thresh, keep_top_k=200)
    #     # print('out', out)
    #     # print('anchors_nms_idx', anchors_nms_idx)
    #     anchors_nms_idx = nms(np.array(paddle.concat(x=[bboxes, scores], axis=2)[0]), nms_thresh)


    #     temp = paddle.to_tensor(cls_score[0][anchors_nms_idx])
    #     nms_scores = paddle.max(temp, axis=1)
    #     nms_class = paddle.argmax(temp, axis=1)
    #     output_boxes = paddle.concat([
    #         paddle.to_tensor(bboxes[0][anchors_nms_idx]),
    #         paddle.to_tensor(anchors[0][anchors_nms_idx])],
    #         axis=1
    #     )
    #     return [nms_scores, nms_class, output_boxes]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2D):
                layer.eval()

    def calc_mining_param(self, process, alpha):
        if process < 0.1:
            bf_weight = 1.0
        elif process > 0.3:
            bf_weight = alpha
        else:
            bf_weight = 5 * (alpha - 1) * process + 1.5 - 0.5 * alpha
        return bf_weight