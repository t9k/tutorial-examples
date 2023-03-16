# 构建自定义镜像

本示例建立并运行一个工作流以在平台上构建自定义镜像并推送到指定的 registry。本示例构建的镜像可以用于 [nanoGPT 项目](https://github.com/karpathy/nanoGPT)的 GPT 模型训练。

切换到当前目录下，先使用 `create-docker-secret.sh` 创建包含身份信息的 Secret（需要提供 registry host、用户名和密码，可选地指定 Secret 的名称），再使用 `workflow.yaml` 创建 WorkflowTemplate：

```shell
# cd into current directory
cd ~/tutorial-examples/build-custom-image
# create a Secret of docker config
# you can provide the username and password via arguments or enter them interactively
./create-docker-secret.sh -r <registry> [-u <username>] [-p <password>] [-s <secret-name>]
# create a WorkflowTemplate
kubectl apply -f workflow.yaml
```

修改 `workflowrun.yaml` 中构建的镜像名称（位于第 14 行）（镜像将被推送到相应的 registry 中，请确保 Secret 包含的身份信息有对应的上传权限），然后使用它创建 WorkflowRun：

```shell
kubectl create -f workflowrun.yaml
```

`workflowrun.yaml` 中的各重要参数如下：

* `contextPath`（9-10 行）：Docker 构建上下文在 PVC 中的路径，这里取 `./tutorial-examples/build-custom-image`。
* `dockerfile`（11-12 行）：dockerfile 在构建上下文中的相对路径，这里取 `Dockerfile`。
* `dstImage`（13-14 行）：构建的镜像名称。
* PVC 名称（23 行）：这里取 `tutorial`。
* Secret 名称（26 行）：这里取 `docker-config`（如果在创建 Secret 时指定了名称，请修改为该名称）。

前往工作流控制台查看镜像的构建日志；构建完成的镜像将被推送到相应的 registry 中。
