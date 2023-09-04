# 教程示例

<div id="top" align="center">

<img src="../assets/illustration.png" alt="illustration" width="300" align="center"><br>

| [English](../README.md) | [中文](README-zh.md) |

</div>

本仓库存放了 TensorStack AI 计算平台的教程示例。

大部分示例都基于“识别 MNIST 手写数字”这一入门级机器学习任务，便于用户理解和上手平台的各项功能。

## 使用方法

1. 在您的项目中创建一个名为 tutorial、大小 4 Gi 以上的 PVC，然后创建一个同样名为 tutorial 的 Notebook 挂载该 PVC，镜像和资源不限（创建 PVC 和 Notebook 的操作步骤请参阅[创建 Notebook](https://t9k.github.io/user-docs/guide/develop-and-test-model/create-notebook.html)；如要使用远程操作，请开启 SSH）。请注意，这里的名称 tutorial 不可以更改，除非您相应地修改示例提供的 YAML 文件。

2. 进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库（使用 Notebook 的方法请参阅[使用 Notebook](https://t9k.github.io/user-docs/guide/develop-and-test-model/use-notebook.html)）。

```shell
cd ~
git clone https://github.com/t9k/tutorial-examples.git
```

3. 继续使用 **Notebook 的终端**，参照各示例的 README 进行操作，或参照各示例在用户文档中对应的教程进行操作。各示例的 README 与相应教程中给出的操作步骤是一致的。

## 示例列表

* 分布式训练：
  * PyTorch 分布式框架：
    * [使用 PyTorchTrainingJob 进行数据并行训练](../job/pytorchtrainingjob/ddp/)
    * [使用 PyTorchTrainingJob 进行参数服务器训练](../job/pytorchtrainingjob/ps/)
  * DeepSpeed：
    * [使用 DeepSpeedJob 进行数据并行训练](../job/deepspeedjob/)
  * ColossalAI：
    * [使用 ColossalAIJob 进行数据并行训练](../job/colossalaijob/)
  * TensorFlow 分布式框架：
    * [使用 TensorFlowTrainingJob 进行数据并行训练](../job/tensorflowtrainingjob/multiworker/)
    * [使用 TensorFlowTrainingJob 进行参数服务器训练](../job/tensorflowtrainingjob/ps/)
  * Horovod：
    * [使用 MPIJob 进行 Horovod 数据并行训练（PyTorch 模型）](../job/mpijob/horovod-torch/)
    * [使用 MPIJob 进行 Horovod 数据并行训练（Keras 模型）](../job/mpijob/horovod-keras/)
  * XGBoost：
    * [使用 XGBoostTrainingJob 进行分布式训练和预测](../job/xgboosttrainingjob/distributed/)
  * Apache Beam：
    * [使用 BeamJob 运行 Apache Beam 分布式计算任务](../job/beamjob/count-word/)
  * 自定义分布式训练：
    * [使用 GenericJob 实现数据并行训练（Keras 模型）](../job/genericjob/multiworker-keras/)
  * [使用 Job 的 debug 模式进行调试](../job/debug-mode/)
* 自动超参数调优：
  * [使用 AutoTune 进行超参数优化（PyTorch 模型）](../autotune/hpo-torch/)
  * [使用 AutoTune 进行超参数优化（Keras 模型）](../autotune/hpo-keras/)
* 记录训练数据和元数据：
  * [单工作器训练使用 EM 记录训练](../em/single-worker-training/)
  * [数据并行训练使用 EM 记录训练](../em/data-parallel-training/)
* 模型部署：
  * 从 PVC 获取模型：
    * [部署用于生产环境的模型推理服务（PyTorch 模型）](../deployment/pvc/mlservice-torch/)
    * [部署用于生产环境的模型推理服务（Keras 模型）](../deployment/pvc/mlservice-keras/)
  * 从 S3 存储获取模型：
    * [部署用于生产环境的模型推理服务（PyTorch 模型，S3 存储）](../deployment/s3/mlservice-torch/)
    * [部署用于生产环境的模型推理服务（Keras 模型，S3 存储）](../deployment/s3/mlservice-keras/)
* 工作流：
  * [建立从数据采样到模型导出的端到端的工作流](../workflow/e2e-workflow/)
* 平台工具：
  * [Codepack 使用示例](../codepack/)
* 其他功能：
  * [在平台上构建镜像](../build-image/build-image-on-platform/)
  * [构建 Notebook 自定义镜像](../build-image/build-notebook-custom-image/)
