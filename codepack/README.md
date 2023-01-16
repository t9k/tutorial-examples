# Codepack 使用示例

本示例通过运行一个简单的 Codepack，来展示 Codepack 的基本功能和使用方法。

<!-- 更多使用 Codepack 机器学习应用请参阅项目 -->

## 操作步骤

1. 切换到当前目录下，运行 Codepack 的 `create-notebook` target：

    ```shell
    # cd into current directory
    cd ~/tutorial-examples/codepack
    codepack run -t create-notebook
    ```

    前往模型构建控制台，可以看到名为 codepack-example 的 PVC 和同名的 Notebook 被创建；进入该 Notebook，可以看到当前目录（连同其下的所有文件）已经被复制到 HOME 目录下。这意味着一个新的专用的模型开发环境已经准备就绪。

2. 运行 `run-distributed-training` target：

    ```shell
    codepack run -t run-distributed-training
    ```

    前往模型构建控制台，可以看到名为 codepack-example 的 TensorFlowTrainingJob 被创建。这意味着模型的分布式训练被启动。

    `run-distributed-training` target 和 `create-notebook` target 共用了两个前置 target，因此在运行过程中会跳过第一个 target（因为 PVC 已经存在），第二个 target 也实际上没有更新 PVC 中的文件（因为具体实现调用的是 `rsync` 命令）。

3. 运行 `clear` target：

    ```shell
    codepack run -t clear
    ```

    前面创建的 PVC、Notebook 和 TensorFlowTrainingJob 都被删除。这意味着运行此 Codepack 而产生的所有改变全部被清理。
