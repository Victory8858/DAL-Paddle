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

import os
import glob
import argparse


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-img-path", default="/home/aistudio/data/DOTA/trainsplit/images", type=str, help="train images path")
    parser.add_argument("--val-img-path", default="/home/aistudio/data/DOTA/valsplit/images", type=str, help="val images path")
    parser.add_argument("--save-dir", default="/home/aistudio/data/DOTA", type=str, help="where to save the file")
    parser.add_argument("--mode", default='trainval', type=str, help="trainval, train, val, test")
    args = parser.parse_args()

    return args


def generate_iamgets(args):
    train_img_path = args.train_img_path
    val_img_path = args.val_img_path
    set_file = os.path.join(args.save_dir, args.mode) + ".txt"

    if args.mode == 'trainval':
        files = sorted(glob.glob(os.path.join(train_img_path, '**.*'))) + sorted(glob.glob(os.path.join(val_img_path, '**.*')))
    elif args.mode == 'train':
        files = sorted(glob.glob(os.path.join(train_img_path, '**.*')))
    elif args.mode == 'val':
        files = sorted(glob.glob(os.path.join(val_img_path, '**.*')))

    with open(set_file, 'w') as f:
        for file in files:
            root_dir = file.split('/images/P')[0]  # root_dir:  /home3/victory8858/dataset/DOTA/train
            label_dir = os.path.join(root_dir, 'labelTxt')  # label_dir:  /home3/victory8858/dataset/DOTA/train\labelTxt
            _, img_name = os.path.split(file)  # P0000__1__0___0.png
            filename = os.path.join(label_dir, img_name[:-4] + '.txt')
            print(filename)
            num_easy = 0
            with open(filename, 'r', encoding='utf-8-sig') as g:
                content = g.read()
                objects = content.split('\n')
                for obj in objects:
                    # print(obj)
                    if len(obj) != 0:
                        *box, class_name, difficult = obj.split(' ')
                        if int(difficult) == 0:   # 有easy样本才加入训练
                            num_easy += 1
                            break
    
            if num_easy > 0:
                img_path, filename = os.path.split(file)
                name, extension = os.path.splitext(filename)
                if extension in ['.jpg', '.bmp', '.png']:
                    f.write(os.path.join(file) + '\n')


if __name__ == '__main__':
    args = getArgs()
    generate_iamgets(args)
