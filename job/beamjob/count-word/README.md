# 使用 BeamJob 运行 Apache Beam 分布式计算任务

本示例使用 BeamJob 通过 [Apache Beam Python SDK](https://beam.apache.org/documentation/sdks/python/) 运行 [Apache Beam](https://beam.apache.org/) 分布式计算任务。

切换到当前目录下，使用 YAML 配置 `job.yaml` 创建 BeamJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/beamjob/count-word/
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态和日志等。
