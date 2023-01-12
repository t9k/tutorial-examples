# 使用 AutoTune 进行超参数优化

本示例使用 AutoTune 对模型进行超参数优化（以 Keras 模型的单机训练为例），并使用 AIMD 服务持久化存储实验数据。

## 准备工作

1. 前往安全管理控制台，创建一个包含完整 AIMD 权限的 API Key，名称任意；然后复制该 API Key。
1. 前往模型构建控制台，创建一个 API Key 类型的 Secret，名称填写 api-key-aimd，API Key 字段粘贴刚才复制的 API Key。

## 操作步骤

1. 前往实验管理控制台，在根路径下创建一个名为 tutorial 的文件夹；然后复制该文件夹的 ID。

1. 回到 Notebook 的终端，切换到当前目录下，编辑文件 `autotune.yaml`，在 `spec.aimd.folder` 字段（第 11 行）粘贴刚才复制的文件夹 ID。

    ```shell
    # cd into current directory
    cd ~/tutorial-examples/autotune/hpo-keras
    vim autotune.yaml
    # paste API Key as value of `spec.aimd.folder` field (line 11)
    ```

1. 使用 `autotune.yaml`（CPU 训练）或 `autotune_gpu.yaml`（GPU 训练，需要 12 个 GPU）创建 AutoTuneExperiment：

    ```shell
    # choose one of the following:
    kubectl create -f autotune.yaml      # CPU training
    kubectl create -f autotune_gpu.yaml  # GPU training
    ```

1. 前往模型构建控制台或实验管理控制台，查看实验的状态、搜索空间，以及各次试验的状态、指标、使用的超参数等详细信息。即使实验被删除，AIMD 服务存储的实验数据依然，因此仍可以在实验管理控制台中访问。
