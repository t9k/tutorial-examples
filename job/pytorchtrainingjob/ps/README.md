# 使用 PyTorchTrainingJob 进行参数服务器训练

本示例演示如何使用 PyTorchTrainingJob 对 PyTorch 模型进行基于 RPC 的参数服务器训练（使用[分布式 RPC 框架 `torch.distributed.rpc`](https://pytorch.org/docs/stable/rpc.html)）。本示例修改自 PyTorch 官方教程 [Implementing a Parameter Server Using Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)，关于训练脚本的更多细节信息请参考此教程。

切换到当前目录下，使用以下 YAML 配置之一创建 PyTorchTrainingJob：

* `job.yaml`：使用 CPU 训练
* `job_gpu.yaml`：使用 GPU 训练
* `job_gpu_t9ksched.yaml`：使用 GPU 训练，并且由 T9k 调度器进行调度（默认的队列名称为 `default`，在 `spec.scheduler.t9kScheduler.queue` 字段（第 8 行）进行修改）

```shell
# cd into current directory
cd ~/tutorial-examples/job/pytorchtrainingjob/ps
# choose one of the following:
# 1. CPU training
kubectl create -f job.yaml
# 2. GPU training
kubectl create -f job_gpu.yaml
# 3. GPU training with Job scheduled by T9k scheduler
# vim job_gpu_t9ksched.yaml  # optionally modify name of queue (line 8)
kubectl create -f job_gpu_t9ksched.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。
