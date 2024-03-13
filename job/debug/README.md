# 调试 Job

本示例使用 Job 的调试模式这一功能来对计算任务进行调试。

debug 模式的设计目标是在保留 Job 原有配置的条件下，方便地对要运行的计算任务进行调试。在 debug 模式下，所有 replica 的容器的启动命令都会被替换为 `["sleep", "inf"]` 或用户指定的其他命令。

debug 模式支持所有类型的 Job，这里以 PyTorchTrainingJob 为例进行演示。代码和 YAML 配置来自示例[使用 PyTorchTrainingJob 进行数据并行训练](../pytorchtrainingjob/ddp/)。

## 容器休眠

如果想要让容器处于休眠状态，可以使用 `sleep` 命令作为启动命令，例如 `["sleep", "3600"]`（休眠 3600 秒，即 1 小时）、`["sleep", "86400"]`（休眠 86400 秒，即 1 天）和 `["sleep", "inf"]`（永久休眠，为默认的启动命令）。用户可以在容器休眠期间手动进入容器，自由地使用各种命令检查环境，或尝试启动训练。

切换到当前目录下，使用 `job_debug.yaml` 创建 PyTorchTrainingJob（YAML 配置文件的第 6-12 行增加了 `spec.runMode` 字段以启用 debug 模式）：

```shell
cd ~/tutorial-examples/job/debug-mode
kubectl create -f job_debug.yaml
```

监控节点的 Pod 的状态直到其变为 `Running`：

```shell
kubectl get pod -l tensorstack.dev/owner-name=torch-mnist-trainingjob-debug -w
```

接下来进入 Pod（的容器）以进行调试：

```shell
kubectl exec -it <POD_NAME> -- bash  # 提供 Pod 的名称
```

## 容器开启 SSH 服务

如果想要让容器开启 SSH 服务，需要先修改镜像，为其安装 SSH 服务器并设置启动选项。当前目录下的 Dockerfile 定义了新的镜像。将执行脚本作为启动命令，随后用户就可以通过 SSH 远程连接 Job（的容器），同样地检查环境或尝试启动训练。

切换到当前目录下，创建一个包含用户 SSH 公钥的 Secret，以及一个包含主机 SSH 密钥对的 Secret：

```shell
cd ~/tutorial-examples/job/debug-mode
./create_ssh_user_key_secret.sh -f <PUBLIC_KEY_FILE>  # 提供用户的 SSH 公钥文件的路径
                                                      # 可以上传 SSH 公钥文件到 Notebook，或创建一个空白文本文件并粘贴公钥
./create_ssh_host_key_secret.sh
```

使用 `job_debug_ssh.yaml` 创建 PyTorchTrainingJob（YAML 配置文件的第 6-12 行增加了 `spec.runMode` 字段以启用 debug 模式）：

```shell
kubectl create -f job_debug_ssh.yaml
```

监控节点的 Pod 的状态直到其变为 `Running`：

```shell
kubectl get pod -l tensorstack.dev/owner-name=torch-mnist-trainingjob-debug-ssh -w
```

然后请参照[此文档](https://t9k.github.io/user-manuals/tasks/ssh-notebook.html)的操作步骤，在本地进行端口转发并建立 SSH 连接。唯一不同的是端口转发使用 `t9k-pf pod` 命令，而非 `t9k-pf notebook` 命令：

```shell
t9k-pf pod <POD_NAME> 5001:2222 -n <PROJECT_NAME>
```

接下来在终端或 IDE 中进行调试，具体取决于用户使用它们中的哪一个进行 SSH 连接。
