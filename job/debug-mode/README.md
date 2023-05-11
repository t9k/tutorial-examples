# 使用 Job 的 debug 模式进行调试

本示例使用 Job 的 debug 模式这一功能对计算任务进行调试。

debug 模式的设计目标是在保留 Job 原有配置的条件下，方便地对要运行的计算任务进行调试。在 debug 模式下，所有 replica 的容器的启动命令都会被替换，用户可以指定新的启动命令（默认为 `["sleep", "inf"]`）。这里以 `sleep inf` 命令进行演示，用户将手动进入容器，自由地使用各种命令检查环境，或尝试启动训练。

目前 debug 模式**仅支持** PyTorchTrainingJob 和 ColossalAIJob，之后将支持所有类型的 Job。这里以 PyTorchTrainingJob 为例进行演示，代码和 YAML 配置来自示例[使用 PyTorchTrainingJob 进行多工作器同步训练](../pytorchtrainingjob/ddp/)，其中 YAML 配置增加了 `spec.runMode` 字段（第 6-12 行）来启用 debug 模式：

<!-- 用户文档添加后移除 -->

```yaml
spec:
  runMode:
    debug:
      enable: true
      replicaSpecs:
        - type: worker
          skipInitContainer: true
          command: ["sleep", "inf"]
```

其中：

* `skipInitContainer`：忽略 replica 的初始化容器，默认为 `false`。
* `command`：replica 的容器的启动命令，默认为 `["sleep", "inf"]`。
* 如果 `spec.runMode.debug.replicaSpecs` 字段未设置某一类 replica，则使用上述默认值；如果未设置 `spec.runMode.debug.replicaSpecs` 字段，则所有 replica 都使用上述默认值。

<!-- 用户文档添加后移除 -->

切换到当前目录下，使用 `job_debug.yaml` 创建 PyTorchTrainingJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/debug-mode
# GPU training with debug mode
kubectl create -f job_debug.yaml
```

在命令行监控作为节点的 Pod `torch-mnist-trainingjob-debug-node-0` 的状态直到变为 `Running`：

```shell
kubectl get pod -w
```

使用一个终端进入 Pod `torch-mnist-trainingjob-debug-node-0`：

```shell
# in one terminal
kubectl exec -it torch-mnist-trainingjob-debug-node-0 -- bash
```

使用 `nvidia-smi` 命令检查当前可用的 GPU，再使用 `ls` 命令检查示例路径是否正确：

```shell
nvidia-smi
ls job/pytorchtrainingjob/ddp
```

然后使用 `torchrun` 命令启动训练：

```shell
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_backend c10d job/pytorchtrainingjob/ddp/torch_mnist_trainingjob.py --log_dir job/pytorchtrainingjob/ddp/log --backend nccl
```

随即分布式训练开始进行。如果训练脚本出错，则可以立即在终端中进行调试，不会造成 Job 的失败。调试完成后禁用 debug 模式（将 `spec.runMode.debug.enable` 设为 `false`，或直接注释第 6-12 行），再次创建 PyTorchTrainingJob 则正常启动训练。
