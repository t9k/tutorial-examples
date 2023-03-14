# 建立从数据采样到模型导出的自动化工作流

本示例建立并运行一个端到端的机器学习工作流，包括数据预处理、模型训练、模型分析、模型上传等步骤。本示例使用的机器学习应用样例是一个二分类问题，根据乘客搭乘出租车的位置、路程、用时等特征数据预测乘客是否会付小费。

## 文件结构

当前目录的文件结构如下：

* `./data`：用于训练、验证、测试的数据集。
* `./src`：数据预处理、模型训练、模型分析、模型上传等每个步骤的 Python 代码，主要使用了 [TFX 库](https://www.tensorflow.org/tfx)。
* `./workflowtemplates`：用于创建 WorkflowTemplate 的 YAML。

## 操作步骤

切换到当前目录下，先使用 `./workflowtemplates` 下的所有文件创建多个 WorkflowTemplate，再使用 `workflowrun.yaml` 创建 WorkflowRun：

```shell
# cd into current directory
cd ~/tutorial-examples/workflow/automatic-workflow
kubectl apply -f workflowtemplates
kubectl create -f workflowrun.yaml
```

前往工作流控制台查看工作流的运行进度以及每个节点的日志；工作流产生的所有文件（包含模型文件、模型分析数据等等）都位于当前目录的 `output` 子目录下。
