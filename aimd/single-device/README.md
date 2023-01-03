# 单设备训练使用 AIMD

本示例使用 AIMD 记录并展示模型在单个设备上进行的一次训练中的各种数据（以 PyTorch 模型在单个 CPU 或 GPU 上的训练为例）。

切换到当前目录下，编辑文件 `job.yaml`，在命令中补全 AIMD 服务器的地址（第 20 行）以及您的 API Key（第 22 行）。

```shell
# cd into current directory
cd ~/tutorial-examples/aimd/single-device
vim job.yaml
# fill in host of AIMD server (line 20) and your API Key (line 22)
# host of AIMD server is like "https://.../t9k/aimd/server"
```

使用 `job.yaml` 创建 PyTorchTrainingJob：

> PyTorchTrainingJob 本身被设计用于运行 PyTorch 的分布式训练，这里为便于演示，将使用只有一个工作器的 PyTorchTrainingJob 来运行这一训练。您也可以在安装了 PyTorch 的本地环境或 Notebook 中直接运行脚本 `torch_mnist_aimd.py`。

```shell
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态和日志等。训练完成后，其数据将被上传到 AIMD 服务器，此时前往实验管理控制台，找到文件夹 aimd-example 下的试验 mnist_torch，进入以查看试验的元数据、运行环境、指标以及超参数。
