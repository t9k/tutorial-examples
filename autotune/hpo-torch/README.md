# 使用 AutoTune 进行超参数优化

本示例使用 AutoTune 对模型进行超参数优化（以 PyTorch 模型的单机训练为例）。

## 操作步骤

1. 切换到当前目录下，使用以下 YAML 配置之一创建 AutoTuneExperiment：

    * `autotune.yaml`：使用 CPU 训练
    * `autotune_gpu.yaml`：使用 GPU 训练（需要 12 个 GPU）
    * `autotune_gpu_t9ksched.yaml`：使用 GPU 训练（需要 12 个 GPU），并且由 T9k 调度器进行调度（默认的队列名称为 `default`，在 `spec.trainingConfig.scheduler.t9kScheduler.queue` 字段（第 20 行）进行修改）

    ```shell
    # cd into current directory
    cd ~/tutorial-examples/autotune/hpo-torch
    # choose one of the following:
    # 1. CPU training
    kubectl create -f autotune.yaml
    # 2. GPU training
    kubectl create -f autotune_gpu.yaml
    # 3. GPU training with all Jobs scheduled by T9k scheduler
    # vim autotune_gpu_t9ksched.yaml  # optionally modify name of queue (line 20)
    kubectl create -f autotune_gpu_t9ksched.yaml
    ```

1. 前往模型构建控制台，查看实验的状态、搜索空间，以及各次试验的状态、指标、使用的超参数等详细信息。
