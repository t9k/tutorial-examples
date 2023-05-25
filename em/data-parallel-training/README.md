# 数据并行训练使用 EM 记录训练

本示例使用 EM 在 Job 分布式训练中记录并展示模型的训练数据（以 PyTorch 模型的数据并行训练为例）。本示例的训练脚本修改自示例[使用 PyTorchTrainingJob 进行数据并行训练](../../job/pytorchtrainingjob/ddp)的训练脚本，在其基础上增加了创建 Run、使用超参数配置模型、记录指标、结束和上传 Run 等步骤。

切换到当前目录下，编辑文件 `job.yaml`，在命令中补全 AIStore 服务器的地址（第 24 行）以及您的 API Key（第 26 行）。

```shell
cd ~/tutorial-examples/em/data-parallel-training
vim job.yaml
# fill in host of AIStore server (line 29) and your API Key (line 31)
```

使用 `job.yaml` 创建 PyTorchTrainingJob：

```shell
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。训练完成后，其数据将被上传到 EM 服务器，此时前往实验管理控制台，找到文件夹 em-example 下的 Run mnist_torch_distributed（文件夹路径和 Run 名称分别被硬编码在 `torch_mnist_trainingjob_em.py` 的第 226 和第 164 行），进入以查看 Run 的元数据、运行环境、指标以及超参数。
