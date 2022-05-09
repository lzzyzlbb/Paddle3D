# 使用文档


## 环境要求

- PaddlePaddle 2.3rc(必须)
- OS 64位操作系统
- Python 3(3.7/3.8/3.9)，64位版本，测试在3.8
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.1
- cuDNN >= 7.6

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA10.2
pip install paddlepaddle_gpu==2.3.0-rc0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

请确保您的PaddlePaddle安装成功。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```
**注意**
1. 如果您希望在多卡环境下使用PaddleDetection，请首先安装NCCL

### 2. 安装paddledet

```
# 克隆仓库
cd <path/to/clone/Paddle3D>
git clone https://github.com/lzzyzlbb/Paddle3D

# 安装其他依赖
cd Paddle3D
pip install -r requirements.txt

```

### 2. 编译算子
```
# install custom iou3d and grid_simple_3d op
cd paddle3d/ops/iou3d_nms/
python setup.py install
cd paddle3d/ops/grid_sample_3d/
python setup.py install
cd ../../../

```

## 使用说明

配置文件在`configs/caddn/`目录下
数据在 data/kitti/training下有一部分，还需要将部分处理好的数据, [kitti_infos_val](https://paddle3d.bj.bcebos.com/caddn/data/kitti_infos_val.pkl), 下载后放到data/kitti/下面


### 评估
```
python evaluate.py --config configs/caddn/caddn.yml --model model_path
```

### 导出静态图模型

导出一整个模型
```
export CUDA_VISIBLE_DEVICES=0
python export.py --config configs/caddn/caddn.yml --model_path model_path
```
- model_path 在 [paddle_caddn](https://paddle3d.bj.bcebos.com/caddn/model/paddle_caddn.pdparams) 需要下载


### 推理

```
python infer.py --cfg configs/caddn/caddn.yml --model_file output/model --output_dir ./infer
```

