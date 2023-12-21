# 在平台上构建镜像

本示例使用 Image Builder 在平台上构建自定义 Docker 镜像并推送到指定的 registry。

## 准备 Dockerfile 文件

这里以构建 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 项目的模型训练镜像为例。阅读该项目的 README 文档，知道模型基于 PyTorch 框架（并且训练可以使用 PyTorch 2.0 的 `compile` 函数加速），项目依赖 `transformers`、`datasets`、`tiktoken`、`wandb` 等 Python 包。于是写出镜像的 Dockerfile 如下：

```dockerfile
# use PyTorch 2.0
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# optional, install packages for infiniband network
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y libibverbs1 librdmacm1 libibumad3 && rm -rf /var/lib/apt/lists/*

# install Python packages
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tiktoken \
    wandb
```

## 创建 Secret

切换到当前目录下，使用 `create-docker-secret.sh` 创建包含身份信息的 Secret（需要提供 registry host、用户名和密码，可选地指定 Secret 的名称）：

```shell
# cd into current directory
cd ~/tutorial-examples/build-image/build-image-on-platform
# create a Secret of docker config
# you can provide the username and password via arguments or enter them interactively
./create-docker-secret.sh -r <registry> [-u <username>] [-p <password>] [-s <secret-name>]
```

## 创建 Image Builder

修改 `image-builder.yaml` 中构建的镜像名称（位于第 11 行）（镜像将被推送到相应的 registry 中，请确保 Secret 包含的身份信息具有相应的上传权限），然后使用它创建 Image Builder：

```shell
kubectl create -f image-builder.yaml
```

`image-builder.yaml` 中的各重要参数如下：

* Image Builder 名称（第 4 行）：这里取 `build-image`（注意不要重名）。
* Secret 名称（第 9 行）：这里取 `docker-config`（如果在创建 Secret 时指定了名称，请修改为该名称）。
* `tag`（第 11 行）：构建的镜像名称。
* `contextPath`（第 14 行）：Docker 构建上下文在 PVC 中的路径，这里取 `./tutorial-examples/build-image/build-image-on-platform`。
* `dockerfilePath`（第 15 行）：dockerfile 在 PVC 中的路径，这里取 `./tutorial-examples/build-image/build-image-on-platform/Dockerfile`。
* PVC 名称（第 16 行）：这里取 `tutorial`。

## 检查构建进度和结果

前往模型构建控制台，查看 Image Builder 的详情和日志；构建完成的镜像将被推送到相应的 registry 中。
