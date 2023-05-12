# 单工作器训练使用 AIMD 记录训练数据

本示例使用 AIMD 记录并展示模型在单个工作器上进行的一次训练中的各种数据（以 PyTorch 模型在单个 CPU 或 GPU 上的训练为例）。本示例的训练脚本修改自示例[使用 PyTorchTrainingJob 进行数据并行训练](../../job/pytorchtrainingjob/ddp)的训练脚本，在其基础上删除了分布式和 TensorBoard 相关的代码，增加了创建试验、使用超参数配置模型、记录指标、结束和上传试验等步骤。

切换到当前目录下，运行 Python 脚本以启动训练（需要提供 AIMD 服务器的地址以及您的 API Key）：

> PyTorchTrainingJob 本身被设计用于运行 PyTorch 的分布式训练，这里为便于演示，将使用只有一个工作器的 PyTorchTrainingJob 来运行这一训练。您也可以在安装了 PyTorch 的本地环境或 Notebook 中直接运行脚本 `torch_mnist_single_aimd.py`。

```bash
python torch_mnist_single_aimd.py --aimd_host <AIMD_SERVER_URL> --api_key <YOUR_API_KEY>
```

训练完成后，其数据将被上传到 AIMD 服务器，此时前往实验管理控制台，找到文件夹 aimd-example 下的试验 mnist_torch（文件夹路径和试验名称被硬编码在 `torch_mnist_single_aimd.py` 的第 132-133 行），进入以查看试验的元数据、运行环境、指标以及超参数。
