# 使用 TensorFlowTrainingJob 进行参数服务器训练

本示例演示如何使用 TensorFlowTrainingJob 对 Keras 模型进行参数服务器（parameter server）训练（采用 [`tf.distribute.experimental.ParameterServerStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy) 分布式策略）。

切换到当前目录下，使用以下 YAML 配置之一创建 TensorFlowTrainingJob：

* `job.yaml`：使用 CPU 训练
* `job_gpu.yaml`：使用 GPU 训练
* `job_gpu_t9ksched.yaml`：使用 GPU 训练，并且由 T9k 调度器进行调度（默认的队列名称为 `default`，在 `spec.scheduler.t9kScheduler.queue` 字段（第 8 行）进行修改）

```shell
# cd into current directory
cd ~/tutorial-examples/job/tensorflowtrainingjob/ps
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
