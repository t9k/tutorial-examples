# 使用 GenericJob 实现 Keras 模型的多工作器同步训练

本示例演示如何使用 GeneribJob 对 Keras 模型进行多工作器同步训练（采用 [`tf.distribute.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy) 分布式策略）。本示例所使用的脚本文件和示例[使用 TensorFlowTrainingJob 进行多工作器同步训练](../../tensorflowtrainingjob/multiworker/)完全相同，GenericJob 的这一 YAML 配置实际上就是 TensorFlowTrainingJob 的具体实现。

切换到当前目录下，使用 `job.yaml` 创建 GenericJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/genericjob/keras-multiworker
kubectl create -f job.yaml
```

前往模型构建控制台查看训练状态和日志等。
