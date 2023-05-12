# 使用 ColossalAIJob 进行数据并行训练

本示例使用 ColossalAIJob 对 PyTorch 模型（PaLM）进行数据并行训练，使用随机数作为训练数据。本示例来自 [ColossalAI 的官方示例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/palm)。

切换到当前目录下，使用 `job.yaml` 创建 ColossalAIJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。

```shell
# cd into current directory
cd ~/tutorial-examples/job/colossalaijob
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态和日志等。
