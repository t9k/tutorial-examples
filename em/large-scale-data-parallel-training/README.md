# 大规模数据并行训练使用 Asset Hub 保存/加载数据集/模型，使用 EM 记录训练

在进行下列操作之前，请先参照相应教程，在当前项目下创建名为 `openwebtext` 的 Dataset Connect。

切换到当前目录下，编辑文件 `job.yaml`，在命令中补全 AIStore 服务器的地址（第 24 行）以及您的 API Key（第 26 行）。

```shell
cd ~/tutorial-examples/em/large-scale-data-parallel-training
vim job.yaml
# fill in host of AIStore server (line 26), host of Asset Hub server (line 27)
# and your API Key (line 31)
```

使用 `job.yaml` 创建 PyTorchTrainingJob：

```shell
kubectl create -f job.yaml
```
