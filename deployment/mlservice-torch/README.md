# 部署用于生产环境的模型推理服务

本示例使用 MLService 部署用于生产环境的模型推理服务（以 PyTorch 模型为例），主要包含以下操作：

1. 将模型归档文件上传到 S3 存储服务
1. 创建 `MLService` 模型推理服务，其使用 S3 中存储的模型
1. 使用命令行向 `MLService` 发送推理请求

关于如何将 PyTorch 自定义训练模型打包为本示例所使用的 torch model archive（.mar）文件，请参阅官方示例 [Digit recognition model with MNIST dataset](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)。

## 准备工作

1. 将集群的 S3 服务对应的 s3cmd 配置文件放置于 `/t9k/mnt/.s3cfg` 路径下，执行 `s3cmd ls` 命令以确认可以访问。
1. 前往模型构建控制台，创建一个 S3-cfg 类型的 Secret，名称填写 s3cfg，服务端点、访问键和密码分别按照上述配置文件中相应的值填写。

## 操作步骤

切换到当前目录下，将 `mnist.mar` 文件同步到 S3 存储的 `s3://tutorial/torch-mnist/` 路径下：

```shell
# cd into current directory
cd ~/tutorial-examples/deployment/mlservice-torch
s3cmd mb s3://tutorial  # if necessary
s3cmd sync mnist.mar s3://tutorial/torch-mnist/
```

使用 `mlservice.yaml` 创建 MLService：

```shell
kubectl create -f mlservice.yaml
```

在命令行监控服务是否准备好：

```shell
kubectl get -f mlservice.yaml -w
```

待其 `READY` 值变为 `true` 后，复制其 `URL` 值。然后测试该推理服务，通过发送如下请求，注意将 `<URL>` 替换为刚才复制的 `URL` 值：

```shell
curl -T test_data/0.png <URL>  # or use `1.png`, `2.png`
```

响应体应是一个类似于下面的 JSON，其预测了图片最有可能是的 5 个数字以及相应的概率：

```json
{
  "0": 1.0,
  "2": 1.3369878815172598e-10,
  "6": 7.102208632401436e-14,
  "5": 5.859716330864836e-14,
  "9": 3.2580891499658536e-15
}
```
