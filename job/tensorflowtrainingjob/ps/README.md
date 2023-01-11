# 使用 TensorFlowTrainingJob 进行参数服务器训练

本示例演示如何使用 TensorFlowTrainingJob 对 Keras 模型进行参数服务器（parameter server）训练（采用 [`tf.distribute.experimental.ParameterServerStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy) 分布式策略）。

切换到当前目录下，使用 `job.yaml`（CPU 训练）或 `job_gpu.yaml`（GPU 训练）创建 TensorFlowTrainingJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/tensorflowtrainingjob/ps
# choose one of the following:
kubectl create -f job.yaml      # CPU training
kubectl create -f job_gpu.yaml  # GPU training
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。
