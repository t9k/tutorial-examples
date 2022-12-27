# 使用 GenericJob 实现 Keras 模型的多工作器同步训练

本示例演示如何使用 GeneribJob 对 Keras 模型进行多工作器同步训练（采用 [`tf.distribute.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy) 分布式策略）。

对比本示例和示例[使用 TensorFlowTrainingJob 进行多工作器同步训练](../../tensorflowtrainingjob/multiworker/)，两者使用相同的脚本文件，运行相同的分布式训练，不同之处仅在于前者使用 GenericJob 而后者使用 TensorFlowTrainingJob。GenericJob 是最基本、通用的作业类型，可以灵活地实现多种计算任务，例如在本示例中，通过在 YAML 配置中暴露每一个工作器的 `2222` 端口服务、为每个工作器设置环境变量 `TF_CONFIG`、设定成功和失败条件从而实现了 TensorFlow 多工作器同步训练的功能。TensorFlowTrainingJob 作为专用于 TensorFlow 分布式框架的作业类型，已经内在地包含了这些配置项，因而用户在使用时更加方便。

切换到当前目录下，使用 `job.yaml` 创建 GenericJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/genericjob/keras-multiworker
kubectl create -f job.yaml
```

前往模型构建控制台查看训练状态和日志等。
