# 部署用于生产环境的模型推理服务（Keras 模型）

本示例使用 MLService 部署用于生产环境的模型推理服务（以 Keras 模型为例），主要包含以下操作：

1. 将训练好保存的模型文件上传到 S3 存储服务
1. 创建 `MLService` 模型推理服务，其使用 S3 中存储的模型
1. 使用命令行向 `MLService` 发送推理请求

## 准备工作

1. 将集群的 S3 服务对应的 s3cmd 配置文件放置于 `/t9k/mnt/.s3cfg` 路径下，执行 `s3cmd ls` 命令以确认可以访问。
1. 前往模型构建控制台，创建一个 S3-cfg 类型的 Secret，名称填写 s3cfg，服务端点、访问键和密码分别按照上述配置文件中相应的值填写。

## 操作步骤

切换到当前目录下，抽取 `saved_model.tar.gz` 文件：

```shell
# cd into current directory
cd ~/tutorial-examples/deployment/mlservice-keras
tar zxvf saved_model.tar.gz
```

得到的 `saved_model` 目录里面是一个在 MNIST 数据集上训练的简易 Keras 模型以 SavedModel 格式保存的文件。

将这些文件同步到 S3 存储的 `s3://tutorial/keras-mnist/` 路径下：

```shell
s3cmd mb s3://tutorial  # if necessary
s3cmd sync saved_model/* s3://tutorial/keras-mnist/
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
curl -H 'content-type:application/json' -d '@data.json' <URL>
```

响应体应是一个类似于下面的 JSON，其预测了数据中包含的 3 张图片分别为数字 0-9 的概率：

```json
{
  "predictions": [
    [
      9.45858192e-13,
      2.10552867e-10,
      1.56355429e-9,
      1.05197406e-8,
      3.80764725e-11,
      2.20902262e-13,
      4.51935573e-15,
      1.0,
      5.78038117e-10,
      6.0478933e-10
    ],
    [
      4.08048491e-11,
      7.25147e-9,
      1.0,
      6.50814165e-16,
      4.15474912e-14,
      1.13560955e-17,
      2.24699274e-12,
      5.74545836e-14,
      1.79823983e-13,
      3.06657856e-17
    ],
    [
      5.25737179e-12,
      0.999999881,
      2.05960116e-9,
      4.00830729e-14,
      9.31844824e-8,
      2.8683e-9,
      2.80698294e-11,
      1.85036839e-8,
      3.0724161e-9,
      6.48676737e-11
    ]
  ]
}
```
