# 使用 Horovod 进行 Keras 模型的多工作器同步训练¶

本示例演示如何使用 MPIJob 对 Keras 模型进行多工作器同步训练（使用 [horovod.tensorflow.keras](https://horovod.readthedocs.io/en/stable/api.html#module-horovod.tensorflow.keras) 模块）。

切换到当前目录下，使用 `job.yaml`（CPU 训练）或 `job_gpu.yaml`（GPU 训练）创建 MPIJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/mpijob/horovod-keras
# choose one of the following:
kubectl create -f job.yaml      # CPU training
kubectl create -f job_gpu.yaml  # GPU training
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。
