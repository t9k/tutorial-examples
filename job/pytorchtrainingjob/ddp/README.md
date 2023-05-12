# 使用 PyTorchTrainingJob 进行数据并行训练

本示例使用 PyTorchTrainingJob 对 PyTorch 模型进行数据并行训练（使用 [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 分布式数据并行模块）。

切换到当前目录下，使用 `job.yaml` 或 `job_cpu.yaml` 创建 PyTorchTrainingJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。

```shell
# cd into current directory
cd ~/tutorial-examples/job/pytorchtrainingjob/ddp
# choose one of the following:
# 1. GPU training
kubectl create -f job.yaml
# 2. CPU training
kubectl create -f job_cpu.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。
