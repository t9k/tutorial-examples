# Job 使用 AIMD

本示例使用 AIMD 在 Job 分布式训练中记录并展示模型的训练数据（以 PyTorch 模型的多工作器同步训练为例）。

切换到当前目录下，编辑文件 `job.yaml`，在命令中补全 AIMD 服务器的地址（第 24 行）以及您的 API Key（第 26 行）。

```shell
# cd into current directory
cd ~/tutorial-examples/aimd/single-device
vim job.yaml
# fill in host of AIMD server (line 24) and your API Key (line 26)
# host of AIMD server is like "https://.../t9k/aimd/server"
```

使用 `job.yaml` 创建 PyTorchTrainingJob：

```shell
kubectl create -f job.yaml
```

在命令行监控训练的运行进度：

```shell
kubectl get -f job.yaml -o wide -w
```

或者前往模型构建控制台查看训练状态、日志和 TensorBoard 等。训练完成后，其数据将被上传到 AIMD 服务器，此时前往实验管理控制台，找到文件夹 aimd-example 下的试验 mnist_torch_distributed，进入以查看试验的元数据、运行环境、指标以及超参数。
