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

from __future__ import print_function
import sys
sys.path.append('/home/aistudio/work/DAL')
# sys.path.append('/home/aistudio/work/DAL/datasets/DOTA_devkit')
import os
import cv2
import paddle
import codecs
import zipfile
import shutil
import argparse
from tqdm import tqdm
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_aabb, rbox_2_quad
from utils.utils import sort_corners, is_image, hyp_parse
from utils.map import eval_mAP

from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge
# from datasets.DOTA_devkit.ResultMerge import ResultMerge
from datasets.DOTA_devkit.dota_evaluation_task1 import task1_eval


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="/home/aistudio/data/DOTA/test", type=str, help="test path")
    parser.add_argument("--weight-path", default="/home/aistudio/weights/model_93.pdparams", type=str, help="model path")
    parser.add_argument("--conf", default=0.1, type=float, help="conf value")
    args = parser.parse_args()

    return args

def dota_evaluate(model,
                  target_size,
                  test_path,
                  conf=0.01):

    root_data, evaldata = os.path.split(test_path)
    splitdata = evaldata + 'split'
    ims_dir = os.path.join(root_data, splitdata + '/' + 'images')
    # ims_dir = os.path.join(root_data, evaldata + '/' + 'images')  # 自己加的
    root_dir = 'outputs'
    res_dir = os.path.join(root_dir, 'detections')  # 裁剪图像的检测结果
    integrated_dir = os.path.join(root_dir, 'integrated')  # 将裁剪图像整合后成15个txt的结果
    merged_dir = os.path.join(root_dir, 'merged')  # 将整合后的结果NMS

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    for f in [res_dir, integrated_dir, merged_dir]:
        if os.path.exists(f):
            shutil.rmtree(f)
        os.makedirs(f)

    ds = DOTADataset()
    # loss = paddle.zeros([3])
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf=conf)
        nt += len(dets)
        out_file = os.path.join(res_dir, im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {} {:.2f}\n'.format(
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                    ds.return_class(dets[k, 0]), im_name[:-4], dets[k, 1], )
                )
    ResultMerge(res_dir, integrated_dir, merged_dir)
    # calc mAP
    mAP, classaps = task1_eval(merged_dir, test_path)
    # display result
    pf = '%20s' + '%10.3g' * 6  # print format    
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    # mAP = 0
    return 0, 0, mAP, 0


def evaluate(target_size,
             test_path,
             dataset,
             backbone=None,
             weight=None,
             model=None,
             hyps=None,
             conf=0.3):
    if model is None:
        model = RetinaNet(backbone=backbone, hyps=hyps)
        if weight.endswith('.pdparams'):
            chkpt = paddle.load(weight)
            # load model
            if 'model' in chkpt.keys():
                model.set_state_dict(chkpt['model'])
                print("Successfully load the model by model!")
            else:
                model.set_state_dict(chkpt)
                print("Successfully load the model!")

    model.eval()

    if dataset == 'DOTA':
        results = dota_evaluate(model, target_size, test_path, conf)
    else:
        raise RuntimeError('Unsupported dataset!')
    return results


if __name__ == '__main__':
    args = getArgs()
    results = evaluate(target_size=[800],
                       test_path=args.test_path,
                       dataset='DOTA',
                       backbone = 'res50',
                       weight=args.weight_path,
                       conf= args.conf)  # p, r, map, f1
