# 使用 AutoTune 进行超参数优化

本示例使用 AutoTune 对模型进行超参数优化（以 PyTorch 模型的单机训练为例）。

## 操作步骤

1. 切换到当前目录下，使用 `autotune.yaml`（CPU 训练）或 `autotune_gpu.yaml`（GPU 训练，需要 12 个 GPU）创建 AutoTuneExperiment：

    ```shell
    # cd into current directory
    cd ~/tutorial-examples/autotune/hpo-torch
    # choose one of the following:
    kubectl create -f autotune.yaml      # CPU training
    kubectl create -f autotune_gpu.yaml  # GPU training
    ```

1. 前往模型构建控制台，查看实验的状态、搜索空间，以及各次试验的状态、指标、使用的超参数等详细信息。
