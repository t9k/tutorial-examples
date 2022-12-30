# 使用 AutoTune 进行超参数优化

本示例使用 AutoTune 对模型进行超参数优化（以 Keras 模型的单机训练为例）。

按照以下步骤进行操作（详细的操作步骤请参阅对应教程）：

1. 前往安全管理控制台，创建一个包含完整 AIMD 权限的 API Key，名称任意；然后复制该 API Key。
1. 前往模型构建控制台，创建一个 API Key 类型的 Secret，名称填写 api-key-aimd，API Key 字段粘贴刚才复制的 API Key。
1. 前往实验管理控制台，在根路径下创建一个名为 example 的文件夹；然后复制该文件夹的 ID。
1. 回到 Notebook 的终端，切换到当前目录下，编辑文件 `autotune.yaml`，在 `spec.aimd.folder` 字段（第 11 行）粘贴刚才复制的文件夹 ID。

    ```shell
    # cd into current directory
    cd ~/tutorial-examples/autotune/hpo
    vim autotune.yaml
    # paste API Key as value of `spec.aimd.folder` field (line 11)
    ```

1. 使用 `autotune.yaml` 创建 AutoTuneExperiment：

    ```shell
    kubectl create -f autotune.yaml
    ```

1. 前往模型构建控制台或实验管理控制台，查看实验的状态、搜索空间，以及各次运行的状态、指标、使用的超参数等详细信息。
