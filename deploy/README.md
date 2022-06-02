## 一. 将模型参数放到caddn_infer_model目录下
模型放好后的目录结构
```
deploy
   ├── caddn_infer_model
   |            ├── model.pdiparams
   |            └── model.pdmodel
   ├── caddn_main.cc
   ...
```

## 二. 修改`compile.sh`

打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-7.0.0.11
```

运行 `bash compile.sh`， 会在目录下产生build目录。


## 三. 运行样例

```shell
# 运行样例
./build/caddn_main
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

> 注：确保路径配置正确后，也可执行执行 `bash run.sh` ，一次性完成以上两个步骤的执行
