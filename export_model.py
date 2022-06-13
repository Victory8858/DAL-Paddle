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
import os
import argparse
import paddle   
from models.model import RetinaNet


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/aistudio/weights/model_92.pdparams", type=str, help="model want to export")
    args = parser.parse_args()

    return args

def export_model(args):
    # 模型参数加载
    model = RetinaNet()
    net_state_dict = paddle.load(args.model_path)
    model.set_dict(net_state_dict['model'])
    print('Load the model paramters successfully!')

    # 模型动转静
    model.eval()
    model = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec(shape=[1, 3, 800, 800], dtype="float32"),
                                                    paddle.static.InputSpec(shape=[1], dtype="float32")])

    # 静态图模型保存
    if not os.path.exists('./export_model'):
        os.mkdir('./export_model')
    paddle.jit.save(model, os.path.join('./export_model', "inference"))


if __name__ == '__main__':
    args = getArgs()
    export_model(args)