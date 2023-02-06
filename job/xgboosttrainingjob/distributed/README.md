# 使用 XGBoostTrainingJob 进行分布式训练和预测

本示例使用 XGBoostTrainingJob 对 XGBoost 模型进行分布式训练和预测。

切换到当前目录下，使用以下 YAML 配置之一创建 XGBoostTrainingJob：

* `job.yaml`：使用 CPU 训练
* `job_gpu.yaml`：使用 GPU 训练
* `job_gpu_t9ksched.yaml`：使用 GPU 训练，并且由 T9k 调度器进行调度（默认的队列名称为 `default`，在 `spec.scheduler.t9kScheduler.queue` 字段（第 8 行）进行修改）

```shell
# cd into current directory
cd ~/tutorial-examples/job/xgboosttrainingjob/distributed/
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

或者前往模型构建控制台查看训练状态和日志等。
