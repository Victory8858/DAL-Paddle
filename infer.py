# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('/home/aistudio/AutoLog')
import os
import cv2
import paddle
from paddle import inference
import numpy as np
from PIL import Image
# from reprod_log import ReprodLogger
from paddle.vision.transforms import Compose, transforms
from utils.utils import Rescale, Normailize, Reshape
from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes
from utils.utils import sort_corners, is_image, hyp_parse
from utils.bbox import rbox_2_aabb, rbox_2_quad

def decoder(ims, anchors, cls_score, bbox_pred, thresh=0.6, nms_thresh=0.2, test_conf=0.1):  # 验证阶段一个一个来的
    ims = paddle.to_tensor(ims)
    anchors = paddle.to_tensor(anchors)
    cls_score = paddle.to_tensor(cls_score)
    bbox_pred = paddle.to_tensor(bbox_pred)
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




class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.

        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.ims, self.anchors_list, self.cls_score, self.bbox_pred, self.test_conf = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.crop_size,
                                   self.args.crop_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor

        initialize the inference engine

        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        ims = predictor.get_output_handle(output_names[0])
        anchors_list = predictor.get_output_handle(output_names[1])
        cls_score = predictor.get_output_handle(output_names[2])
        bbox_pred = predictor.get_output_handle(output_names[3])
        test_conf = predictor.get_output_handle(output_names[4])

        return predictor, config, input_tensor, ims, anchors_list, cls_score, bbox_pred, test_conf

    def preprocess(self, img_path):
        """preprocess

        Preprocess to the input.

        Args:
            img_path: Image path.

        Returns: Input data after preprocess.
        """
        raw = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 与原版不一样
        img, im_scales = Rescale(target_size=800, keep_ratio=True)(raw)
        img = Compose([Normailize(), Reshape(unsqueeze=True)])(img)
        return img, im_scales, raw

    def postprocess(self, scores, classes, boxes, im_scales):
        """postprocess

        Postprocess to the inference engine output.

        Args:
            x: Inference engine output.

        Returns: Output data after argmax.
        """

        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        boxes[:, :4] = boxes[:, :4] / im_scales
        if boxes.shape[1] > 5:
            boxes[:, 5:9] = boxes[:, 5:9] / im_scales
        scores = np.reshape(scores, (-1, 1))
        classes = np.reshape(classes, (-1, 1))
        cls_dets = np.concatenate([classes, scores, boxes], axis=1)
        keep = np.where(classes > 0)[0]
        dets = cls_dets[keep, :]
        res = sort_corners(rbox_2_quad(dets[:, 2:]))
        polys = []
        for k in range(dets.shape[0]):
            # print('res', res)
            poly = []
            cur_det = res[k] 
            for i in range(4):
                pair = []
                pair.append(int(cur_det[i*2]))
                pair.append(int(cur_det[i*2 + 1]))
                # poly.append(tuple(pair))
                poly.append(pair)
            polys.append(poly)
        return polys


    def run(self, x):
        """run

        Inference process using inference engine.

        Args:
            x: Input data after preprocess.

        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()

        ims = self.ims.copy_to_cpu()
        anchors_list = self.anchors_list.copy_to_cpu()
        cls_score = self.cls_score.copy_to_cpu()
        bbox_pred = self.bbox_pred.copy_to_cpu()
        test_conf = self.test_conf

        nms_scores, nms_class, output_boxes = decoder(ims, anchors_list, cls_score, bbox_pred, test_conf)
        # cores, classes, boxes
        return nms_scores, nms_class, output_boxes


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(description="PaddlePaddle Detection", add_help=add_help)

    parser.add_argument("--model-dir", default='/home/aistudio/work/DAL/export_model', help="inference model dir")
    parser.add_argument("--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--max-batch-size", default=8, type=int, help="max_batch_size")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--img-path", default="/home/aistudio/work/DAL/tiny_datasetsplit/images/P0000__1__2400___3600.png")
    parser.add_argument("--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")
    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args: 
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="detection",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img, im_scales, raw = inference_engine.preprocess(args.img_path)

    # run
    scores, classes, boxes = inference_engine.run(np.array(img))

    # postprocess
    polys = inference_engine.postprocess(scores, classes, boxes, im_scales) 

    for poly in polys:
        imgRet = cv2.polylines(raw, np.array([poly]), True, (0, 255, 0), 3)
    save_path = './output_img'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, "output_img.png"), imgRet)

    if args.benchmark:
        autolog.times.stamp()

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    return polys


if __name__ == "__main__":
    args = get_args()
    polys = infer_main(args)
    print('Predicted Boxes: ', polys)