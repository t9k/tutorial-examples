# 使用 DeepSpeedJob 进行数据并行训练

本示例使用 DeepSpeedJob 对 PyTorch 模型进行数据并行训练，使用 CIFAR-10 作为训练数据。本示例来自 [DeepSpeed 的官方示例](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar)。

切换到当前目录下，使用 `job.yaml` 创建 DeepSpeedJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。

```shell
# cd into current directory
cd ~/tutorial-examples/job/deepspeedjob
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job_gpu.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态和日志等。
