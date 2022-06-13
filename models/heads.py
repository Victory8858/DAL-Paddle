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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CLSHead(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_classes):
        super(CLSHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.convs = nn.LayerList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2D(chns, feat_channels, 3, 1, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0, math.sqrt(2. / (9*feat_channels))))))
            self.convs.append(nn.ReLU())
        self.head = nn.Conv2D(feat_channels, num_anchors * num_classes, 3, 1, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0)), bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant((-math.log((1.0 - 0.01) / 0.01)))))

    def forward(self, x):
        for conv in self.convs:
            # print('x.shape', x.shape)
            # print('conv', conv)
            x = conv(x)   # 这里有问题啊
        x = F.sigmoid(self.head(x))
        x = paddle.transpose(x, [0, 2, 3, 1])  # x = x.permute(0, 2, 3, 1)
        n, w, h, c = x.shape
        x = paddle.reshape(x, shape=[n, w, h, self.num_anchors, self.num_classes])  # x = x.reshape(n, w, h, self.num_anchors, self.num_classes)

        return paddle.reshape(x, shape=[x.shape[0], -1, self.num_classes])


class REGHead(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_regress = num_regress
        self.convs = nn.LayerList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2D(chns, feat_channels, 3, 1, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0, math.sqrt(2. / (9*feat_channels))))))
            self.convs.append(nn.ReLU())
        self.head = nn.Conv2D(feat_channels, num_anchors * num_regress, 3, 1, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0)), bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0)))
        # self.head = nn.Conv2D(feat_channels, num_anchors * num_regress, 3, 1, 1)
        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2D):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2D):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #     self.head.weight.data.fill_(0)
    #     self.head.bias.data.fill_(0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.head(x)
        x = paddle.transpose(x, [0, 2, 3, 1])  # x = x.permute(0, 2, 3, 1)

        return paddle.reshape(x, shape=[x.shape[0], -1, self.num_regress])
        # return x.reshape(x.shape[0], -1, self.num_regress)



if __name__ == '__main__':
    import numpy as np

    feature = np.random.random([2, 256, 100, 100])
    cls = CLSHead(in_channels=256, feat_channels=256, num_stacked=4, num_anchors=3, num_classes=16)