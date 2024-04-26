# 从 Hugging Face 下载模型和数据集到 Asset Hub

本示例使用工作流将 Hugging Face 的模型下载并存储到 Asset Hub。

## 操作步骤

创建一个包含用户 T9k Config 的 Secret。生成 T9k Config 的方法请参阅 [T9k CLI 的配置文件](https://t9k.github.io/user-manuals/latest/tools/cli-t9k/guide.html#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)；创建 Secret 的方法请参阅[创建 Secret](https://t9k.github.io/user-manuals/latest/tasks/manage-secret.html#%E5%88%9B%E5%BB%BA-secret)，其中类型选择 **Custom**，添加一个数据：键为 `.t9kconfig`，值为 T9k Config。

切换到当前目录下，先使用 `./workflowtemplates` 下的所有文件创建多个 WorkflowTemplate，再使用 `workflowrun.yaml` 创建 WorkflowRun：

```shell
# cd into current directory
cd ~/tutorial-examples/workflow/hf-to-ah
kubectl apply -f workflowtemplates
kubectl create -f workflowrun.yaml
```

前往工作流控制台查看 WorkflowRun 的日志。
