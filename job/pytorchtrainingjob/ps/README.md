# 使用 PyTorchTrainingJob 进行参数服务器训练

本示例使用 PyTorchTrainingJob 对 PyTorch 模型进行基于 RPC 的参数服务器训练（使用[分布式 RPC 框架 `torch.distributed.rpc`](https://pytorch.org/docs/stable/rpc.html)）。本示例修改自 PyTorch 官方教程 [Implementing a Parameter Server Using Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)，关于训练脚本的更多细节信息请参考此教程。

切换到当前目录下，使用 `job.yaml` 或 `job_cpu.yaml` 创建 PyTorchTrainingJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。

```shell
# cd into current directory
cd ~/tutorial-examples/job/pytorchtrainingjob/ps
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
