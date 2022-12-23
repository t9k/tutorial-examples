# 使用 Horovod 进行 Keras 模型的多工作器同步训练¶

本示例演示如何使用 MPIJob 对 Keras 模型进行多工作器同步训练（使用 [horovod.tensorflow.keras](https://horovod.readthedocs.io/en/stable/api.html#module-horovod.tensorflow.keras) 模块）。

切换到当前目录下，使用 `job.yaml` 创建 MPIJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/mpijob/horovod-keras
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。
