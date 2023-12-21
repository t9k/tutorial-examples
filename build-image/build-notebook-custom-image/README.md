# 构建 Notebook 自定义镜像

本示例使用 Image Builder 在平台上构建 Notebook 的自定义镜像并推送到指定的 registry。

运行本示例之前，请先完成示例[在平台上构建镜像](../build-image-on-platform)。

## 准备工作

参照示例*在平台上构建镜像*的[创建 Secret](../build-image-on-platform/README.md#创建-secret)部分，创建包含身份信息的 Secret。

## 修改 Notebook 标准镜像

第一种方法直接在标准镜像的基础上进行修改，更加简单方便。

### 准备 Dockerfile 文件

例如我们要在 `t9kpublic/torch-2.1.0-notebook:1.77.1` 这一标准镜像的基础上增加文件、安装 Python 包和 Debian 软件包，于是写出镜像的 Dockerfile 如下（即 `Dockerfile.patched` 文件）：

```dockerfile
FROM t9kpublic/torch-2.1.0-notebook:1.77.1

USER root
WORKDIR /t9k/export

# copy files
COPY . .

# install Python packages, e.g. tiktoken
RUN pip install --no-cache-dir -r ./requirements.txt

# install Debian packages, e.g. iputils-ping
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y iputils-ping && rm -rf /var/lib/apt/lists/*

USER t9kuser
WORKDIR /t9k/mnt
```

### 创建 Image Builder

切换到当前目录下，修改 `image-builder.patched.yaml` 中构建的镜像名称（位于第 11 行）（镜像将被推送到相应的 registry 中，请确保 Secret 包含的身份信息具有相应的上传权限），然后使用它创建 Image Builder：

```shell
# cd into current directory
cd ~/tutorial-examples/build-image/build-notebook-custom-image
# create an Image Builder
kubectl create -f image-builder.patched.yaml
```

## 从零开始构建 Notebook 自定义镜像

第二种方法从零开始构建镜像，自定义程度高，但更加复杂，并且花费的时间更长。

### 准备 Dockerfile 文件

为便于演示说明，这里构建与第一种方法相同的镜像。`Dockerfile.full` 文件给出了完整的构建指令，其中：

* 第 7 行指定了基础镜像，这里为 PyTorch 官方镜像 `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel`。
* 第 45-46 行指定了新安装的 Debian 软件包。
* 第 74-76 行指定了要复制的文件。
* 第 99-100 行指定了新安装的 Python 包。

> 如果您想要从零开始构建自己的 Notebook 自定义镜像（PyTorch 1/2 环境），推荐基于 `Dockerfile.full` 文件，并对上述说明的各行进行相应的修改。
> 
> 如果想要构建 Notebook 自定义镜像（TensorFlow 2 环境），推荐基于 `Dockerfile.full.tf2` 文件进行修改。

### 运行工作流

切换到当前目录下，修改 `workflowrun.full.yaml` 中构建的镜像名称（位于第 12 行）（镜像将被推送到相应的 registry 中，请确保 Secret 包含的身份信息具有相应的上传权限），然后使用它创建 WorkflowRun：

```shell
# cd into current directory
cd ~/tutorial-examples/build-image/build-notebook-custom-image
# create a WorkflowRun
kubectl create -f workflowrun.full.yaml
```

## 检查构建进度和结果

前往工作流控制台查看镜像的构建日志；构建完成的镜像将被推送到相应的 registry 中。

前往模型构建控制台使用构建的自定义镜像创建 Notebook，进入 Notebook 检查增加的文件和安装的 Python 包、Debian 软件包是否存在。
