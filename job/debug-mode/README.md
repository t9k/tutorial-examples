# 使用 Job 的 debug 模式进行调试

本示例使用 Job 的 debug 模式这一功能对要运行的计算任务进行调试。

在 debug 模式下，所有 replica 的容器的启动命令都会被替换为 `sleep inf`，这时由用户进入容器执行任意命令以进行调试。用户可以自由地使用各种命令检查环境，或尝试启动训练（这里训练脚本出错不会造成 Job 的失败）。

注意：目前 debug 模式**仅支持** PyTorchTrainingJob 和 ColossalAIJob，之后将支持所有类型的 Job。这里以 PyTorchTrainingJob 为例进行演示，代码和 YAML 配置来自示例[使用 PyTorchTrainingJob 进行多工作器同步训练](../pytorchtrainingjob/ddp/)，其中 YAML 配置增加了 `spec.runMode` 字段（第 6-11 行）来启用 debug 模式。

切换到当前目录下，使用 `job_debug.yaml` 创建 PyTorchTrainingJob：

```shell
# cd into current directory
cd ~/tutorial-examples/job/debug-mode
# CPU training with debug mode
kubectl create -f job_debug.yaml
```

在命令行监控作为工作器的 Pod `torch-mnist-trainingjob-debug-worker-0` 和 `torch-mnist-trainingjob-debug-worker-1` 的状态直到变为 `Running`：

```shell
kubectl get pod -w
```

然后使用两个终端分别进入这两个 Pod，

```shell
# in one terminal
kubectl exec -it torch-mnist-trainingjob-debug-worker-0 -- bash
python job/pytorchtrainingjob/ddp/torch_mnist_trainingjob.py --log_dir job/pytorchtrainingjob/ddp/log --backend gloo --no_cuda
```

```shell
# in another terminal
kubectl exec -it torch-mnist-trainingjob-debug-worker-1 -- bash
python job/pytorchtrainingjob/ddp/torch_mnist_trainingjob.py --log_dir job/pytorchtrainingjob/ddp/log --backend gloo --no_cuda
```

如果训练脚本出错，则可以立即进行调试。调试完成后禁用 debug 模式（将 `spec.runMode.debug.enable` 设为 `false`，或直接注释第 6-11 行），再次创建 PyTorchTrainingJob 则正常启动训练。
