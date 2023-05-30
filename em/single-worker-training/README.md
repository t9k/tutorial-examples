# 单工作器训练使用 EM 记录训练

本示例使用 EM 记录并展示模型在单个工作器上进行的一次训练中的各种数据（以 PyTorch 模型在单个 CPU 或 GPU 上的训练为例）。本示例的训练脚本修改自示例[使用 PyTorchTrainingJob 进行数据并行训练](../../job/pytorchtrainingjob/ddp)的训练脚本，在其基础上删除了分布式和 TensorBoard 相关的代码，增加了创建 Run、使用超参数配置模型、记录指标、结束和上传 Run 等步骤。

切换到当前目录下，运行 Python 脚本以启动训练（本地环境需要安装 PyTorch 以及 [TensorStack SDK](https://t9k.github.io/user-docs/tool/tensorstack-sdk/index.html)，并具有[配置文件](https://t9k.github.io/user-docs/tool/tensorstack-sdk/user-guide.html#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)）：

```shell
cd ~/tutorial-examples/em/single-worker-training
python torch_mnist_single_em.py
```

训练完成后，其数据将被上传到 EM 服务器，此时前往实验管理控制台，找到文件夹 em-example 下的 Run mnist_torch（文件夹路径和 Run 名称被硬编码在 `torch_mnist_single_em.py` 的第 132-133 行），进入以查看 Run 的元数据、运行环境、指标以及超参数。
