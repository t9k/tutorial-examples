# 从 Hugging Face 下载模型和数据集到 Asset Hub

本示例使用工作流将 Hugging Face 的模型下载并存储到 Asset Hub。数据集同理。

## 操作步骤

创建一个名为 t9k-config、包含用户 T9k Config 的 Secret。生成 T9k Config 的方法请参阅 [T9k CLI 的配置文件](https://t9k.github.io/user-manuals/latest/tools/cli-t9k/guide.html#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)；创建 Secret 的方法请参阅[创建 Secret](https://t9k.github.io/user-manuals/latest/tasks/manage-secret.html#%E5%88%9B%E5%BB%BA-secret)，其中类型选择 Custom，添加一个数据：键为 `.t9kconfig`，值为 T9k Config。

切换到当前目录下，先使用 `./workflowtemplates` 下的所有文件创建多个 WorkflowTemplate，再使用 `workflowrun.yaml` 创建 WorkflowRun：

```shell
# cd into current directory
cd ~/tutorial-examples/workflow/hf-to-ah
kubectl apply -f workflowtemplate.yaml
kubectl create -f workflowrun.yaml
```

对于 `workflowrun.yaml` 中提供的参数进行如下说明：

* `name`：从 Hugging Face 或 ModelScope 下载的模型或数据集的名称。
* `folder`：Asset Hub 中的文件夹路径，下载的模型或数据集将被放置在这里。
* `token`：Hugging Face token，用于登录到 Hugging Face 以下载受保护模型或数据集。请参阅 [User access tokens](https://huggingface.co/docs/hub/en/security-tokens)。
* `source`：下载源。如果从 Hugging Face 下载，将此参数设为 `hf`；如果从 ModelScope 下载，将此参数设为 `modelscope`。
* `httpProxy`：可选的 HTTP 和 HTTPS 代理。

前往工作流控制台查看 WorkflowRun 的日志。待 WorkflowRun 运行完成之后，进入 Asset Hub 控制台，可以看到下载的模型已经被放置在指定的文件夹中。
