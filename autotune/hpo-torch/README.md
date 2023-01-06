# 使用 AutoTune 进行超参数优化

本示例使用 AutoTune 对模型进行超参数优化（以 PyTorch 模型的单机训练为例）。

## 操作步骤

1. 切换到当前目录下，使用 `autotune.yaml` 创建 AutoTuneExperiment：

    ```shell
    # cd into current directory
    cd ~/tutorial-examples/autotune/hpo-torch
    kubectl create -f autotune.yaml
    ```

1. 前往模型构建控制台，查看实验的状态、搜索空间，以及各次试验的状态、指标、使用的超参数等详细信息。
