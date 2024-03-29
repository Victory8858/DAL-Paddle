# Linux端基础训练推理功能测试

Linux端基础训练推理功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
| :------: | :------: | :------: | :------: | :------: | :------------------: |
|  DAL  |  DAL  | 正常训练 |    -     |    -     |          -           |


- 推理相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的推理功能汇总如下，

| 算法名称 | 模型名称 | 模型类型 | device | batchsize | tensorrt | mkldnn | cpu多线程 |
| :------: | :------: | -------- | :----: | :-------: | :------: | :----: | :-------: |
|  DAL  |  DAL  | 正常模型 |  GPU   |     1     |    -     |   -    |     -     |


## 2. 测试流程

### 2.1 准备数据

少量数据在`tiny_dataset`与`tiny_datasetsplit`下

### 2.2 准备环境


- 安装PaddlePaddle >= 2.2
- 安装AutoLog（规范化日志输出工具）
    ```
    pip install git+https://hub.fastgit.org/LDOUBLEV/AutoLog
    ```

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_whole_infer
```

以DAL的`Linux GPU/CPU 基础训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/DAL/train_infer_python.txt  lite_train_lite_infer
```

log中输出结果如下，表示命令运行成功。

```
Run successfully with command - python train.py --train-path="tiny_datasetsplit/train.txt" --test-path="tiny_dataset" --save-path="log" --ckpt-path="weights" --test_interval=4 --save_interval=3 --ckpt-path=log/DAL/lite_train_lite_infer/norm_train_gpus_0 --epochs=4   --batch-size=4!  
Run successfully with command - python val.py --test-path="tiny_dataset" --weight-path="log/DAL/lite_train_lite_infer/norm_train_gpus_0/last.pdparams"  !
Run successfully with command - python export_model.py --model-path="/home/aistudio/weights/model_92.pdparams"    ! 
Run successfully with command - python infer.py --img-path="tiny_datasetsplit/images/P0000__1__2400___3600.png" --use-gpu=True --benchmark=log/DAL/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1     > log/DAL/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 ! 
```
TIPC结果：

<div align="center">
    <img src="./1.png" width=800">
</div>

<div align="center">
    <img src="./2.png" width=800">
</div>
## 3. 更多教程

本文档为功能测试用，更丰富的训练预测使用教程请参考：  

* [模型训练、预测、推理教程](../../README.md)  