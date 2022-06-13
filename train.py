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
import os
sys.path.append('/home/aistudio/external-libraries')
sys.path.append('/home/aistudio/work/DAL')
sys.path.append('/home/aistudio/work/DAL/datasets/DOTA_devkit')
import argparse
from datasets import *
from utils.utils import *
import paddle
from paddle.io import DataLoader
from tqdm import tqdm
import glob
import logging
from paddle.optimizer.lr import LRScheduler
from models.model import RetinaNet
from val import evaluate


class DAL_LR(LRScheduler):
    def __init__(self, verbose=False):
        self.lr_list = [0.0001, 9.999999999999999e-06, 1.8594235253127368e-05, 4.1094235253127366e-05, 6.890576474687263e-05, 9.140576474687264e-05,
                        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 
                        0.0001, 0.0001, 1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05, 
                        1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-06,  1e-06, 
                        1e-06,  1e-06,  1e-06,  1e-06,  1e-06,  1e-06]
        self.num = 0
        super(DAL_LR, self).__init__(verbose)

    def get_lr(self):
        lr = self.lr_list[self.num]
        self.num += 1
        return lr

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="/home/aistudio/data/DOTA/trainval.txt", type=str, help="train images path")
    parser.add_argument("--test-path", default="/home/aistudio/data/DOTA/val", type=str, help="train images path")
    parser.add_argument("--save-path", default="/home/aistudio/log", type=str, help="where to save the log")
    parser.add_argument("--ckpt-path", default="/home/aistudio/work/DAL/weights", type=str, help="where to save the log")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--test_interval", default=2, type=int)
    parser.add_argument("--save_interval", default=5, type=int)
    args = parser.parse_args()

    return args


def train(args):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(
        filename=os.path.join(save_path, 'train.log'), filemode='a',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)

    hyps = hyp_parse('/home/aistudio/work/DAL/hyp.py')
    print(hyps)
    # logging.info('Hyper Paramters: {}'.format(hyps))
    train_path = args.train_path
    batch_size = 8
    logging.info('batch_size:{}'.format(batch_size))
    train_set = DOTADataset(dataset=train_path, augment=True)
    collater = Collater(scales=800, keep_ratio=True, multiple=32)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, collate_fn=collater, shuffle=True, drop_last=True)
    logging.info('Train path:{}  batch size:{}'.format(train_path, batch_size))

    #  parse configs
    epochs = args.epochs
    batch_size = args.batch_size
    results_file = 'result.txt'
    start_epoch = 0
    best_fitness = 0
    # creat folder
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    for f in glob.glob(results_file):
        os.remove(f)
    init_seeds()

    model = RetinaNet()
    # scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.0001, milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    # scheduler = CosineWarmup(lr=1e-04, step_each_epoch=1, epochs=100, end_lr=1e-04, verbose=True)
    scheduler = DAL_LR()
    clip = paddle.nn.ClipGradByNorm(clip_norm=0.1)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(), grad_clip=clip)
    # optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    model_info(model, report='summary')  # 'full' or 'summary'
    results = (0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1'
    best_fitness = 0  # max f1

    # weight = '/home/aistudio/weights' + os.sep + 'last.pdparams'
    # opt = '/home/aistudio/weights' + os.sep + 'last.pdopt'
    # # load chkpt
    # if weight.endswith('.pdparams'):
    #     chkpt = paddle.load(weight)
    #     # load model
    #     if 'model' in chkpt.keys() :
    #         model.set_state_dict(chkpt['model'])
    #     else:
    #         model.set_state_dict(chkpt)
    #     # load optimizer
    #     if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None:
    #         opt_dict = paddle.load(opt)
    #         optimizer.set_state_dict(opt_dict)
    #         best_fitness = chkpt['best_fitness']
    #     # load results
    #     if 'training_results' in chkpt.keys() and  chkpt.get('training_results') is not None:
    #         with open(results_file, 'w') as file:
    #             file.write(chkpt['training_results'])  # write results.txt
    #     if 'epoch' in chkpt.keys():
    #         start_epoch = chkpt['epoch'] + 1
    #         for i in range(int(start_epoch)):
    #             scheduler.step()
    #         print("restart from epoch:", start_epoch)
            
    #     del chkpt


    for epoch in range(start_epoch, epochs):
        print(('\n' + '%10s' * 7) % ('Epoch', 'lr', 'cls', 'reg', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar
        mloss = paddle.zeros([2])
        for i, (ni, batch) in enumerate(pbar):
            model.train()
            optimizer.clear_grad()
            ims, gt_boxes = batch['image'], batch['boxes']
            if(gt_boxes.shape[1] == 0):
                # print('img_path:', img_path)
                continue
            losses = model(ims, gt_boxes=gt_boxes, process=epoch/epochs)
            loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
            loss = loss_cls + loss_reg
            if not paddle.isfinite(loss):
                import ipdb
                ipdb.set_trace()
                print('WARNING: non-finite loss, ending training')
                break
            if bool(loss == 0 or loss == 0.):
                continue
            loss.backward()
            optimizer.step()
            loss_items = paddle.stack([loss_cls, loss_reg], 0).squeeze().detach()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            s = ('%10s' + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), optimizer.get_lr(), *mloss, mloss.sum(), gt_boxes.shape[1], min(ims.shape[2:]))
            pbar.set_description(s)
        scheduler.step()
        final_epoch = epoch + 1 == epochs

        # eval
        if args.test_interval != -1 and epoch % args.test_interval == 0 and epoch >= 1:
            # if paddle.cuda.device_count() > 1:
            #     results = evaluate(target_size=[800],
            #                        test_path=args.test_path,
            #                        dataset='DOTA',
            #                        model=model.module,
            #                        hyps=hyps,
            #                        conf=0.01 if final_epoch else 0.1)
            # else:
            results = evaluate(target_size=[800],
                                test_path=args.test_path,
                                dataset='DOTA',
                                model=model,
                                hyps=hyps,
                                conf=0.01 if final_epoch else 0.1)  # p, r, map, f1

        # Write result log
        with open('/home/aistudio/work/DAL/result.txt', 'a') as f:
            f.write(s + '%10.3g' * 4 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        logging.info(s + '%10.3g' * 4 % results + '\n')

        # Checkpoint
        fitness = results[-2]  # Update best mAP
        if fitness > best_fitness:
            best_fitness = fitness

        with open('/home/aistudio/work/DAL/result.txt', 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': f.read(),
                    'model': model.state_dict()}
                    # 'optimizer': None if final_epoch else optimizer.state_dict()}

        
        # model_name = args.ckpt_path + 'model_' + str(epoch) + '.pdparams'
        # opt_name = args.ckpt_path + 'opt_' + str(epoch) + '.pdopt'
        # paddle.save(chkpt, model_name)
        # paddle.save(optimizer.state_dict(), opt_name)

        # Save last checkpoint
        paddle.save(chkpt, os.path.join(args.ckpt_path, 'last.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(args.ckpt_path, 'last.pdopt'))

        # Save best checkpoint
        if best_fitness == fitness:
            paddle.save(chkpt, os.path.join(args.ckpt_path, 'best.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(args.ckpt_path, 'best.pdopt'))

        if (epoch % args.save_interval == 0 and epoch > 100) or final_epoch:
            # if paddle.cuda.device_count() > 1:
            #     paddle.save(chkpt, './weights/deploy%g.pdparams' % epoch)
            # else:
            paddle.save(chkpt, os.path.join(args.ckpt_path, 'deploy%g.pdparams' % epoch))  

if __name__ == '__main__':
    args = getArgs()
    train(args)  