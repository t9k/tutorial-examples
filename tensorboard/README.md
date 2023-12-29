# 使用 TensorBoard 可视化模型训练过程

本示例使用 TensorBoard 可视化模型训练过程。

除了本示例的使用方法外，您也可以在创建 TrainingJob 时，同时创建 TensorBoard 对训练过程和结果进行实时可视化。例如在示例[使用 TensorFlowTrainingJob 进行数据并行训练](https://github.com/t9k/tutorial-examples/tree/master/job/tensorflowtrainingjob/multiworker)中，`job.yaml` 文件设置了 `tensorboardspec` 字段来创建 TensorBoard。

切换到当前目录下，提取 `log.tar.gz` 中保存的训练日志：

```shell
# cd into current directory
cd ~/tutorial-examples/tensorboard
tar zxvf log.tar.gz
```

得到的 `log` 目录里面是一个在 MNIST 数据集上训练简易 Keras 模型时的日志。

使用 `tensorboard.yaml` 配置创建 TensorBoard：

```shell
kubectl create -f tensorboard.yaml
```

查看 TensorBoard 的创建情况：

```shell
kubectl get -f tensorboard.yaml -o wide -w
```

TensorBoard 的 PHASE 变为 `Running` 后，可以前往模型构建控制台查看 TensorBoard。
