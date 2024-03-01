# Transformer Example

使用 MLService Transformer 的完整示例，详情如下：

* 模型功能：图片分类。
* 模型框架：tensorflow
* 数据集：keras.datasets.fashion_mnist
* 推理服务：通过 MLService 部署推理服务，向推理服务发送图片即可获得预测结果。

## 部署服务

切换到当前目录下，使用 `runtime.yaml` 创建 MLServiceRuntime：

```shell
cd ~/tutorial-examples/deployment/mlservice/transformer
kubectl create -f runtime.yaml
```


使用 `mlservice.yaml` 创建 MLService：

``` shell
kubectl apply -f ./mlservice.yaml
```

## Transformer 说明

上述部署的 MLService 使用了提前制定好的 Transformer Image，这个 Transformer 的作用是：
1. 将用户发送的图片转换为 JSON 格式的数据，然后再发送到 Predictor。
2. 将 Predictor 返回的预测结果转换为可读形式。

制作 Transformer Image 请参考[制作 Transformer 镜像](#制作-transformer-镜像)章节。


## 测试服务

查看 MLService 状态，并等待 `Ready` 一栏变为 `True`

``` bash
kubectl get -f ./mlservice.yaml -w
```

待其 `READY` 值变为 `true` 后，通过发送如下请求测试该推理服务：

```
address=$(kubectl get -f mlservice.yaml -ojsonpath='{.status.address.url}')
curl --data-binary @./shoe.png ${address}/v1/models/model:predict
```

响应体应是一个类似于下面的 JSON，其预测了图片最有可能的种类：

```json
{"predictions": ["Ankle boot"]}
```

## 制作 Transformer 镜像

本小节演示如何使用 TensorStack SDK 制作 Transformer 镜像。

### 编写 Transformer 逻辑

在当前目录下的 [server.py](./server.py) 文件中，通过继承 TensorStack SDK 中的 `MLTransformer` 并重写 `preprocess` 和 `postprocess` 方法实现了一个 Transformer：

* `preprocess`：预处理函数，Transformer 收到用户发送的数据，使用 `preprocess` 对数据进行处理，然后再发送给推理服务。在这个示例中，先转换输入图片的数据格式，需要保持与训练的模型的输入数据一致，然后再转换为推理服务的输入格式。
* `postprocess`：后处理函数，Transformer 收到推理服务返回的结果，使用 `postprocess` 对其进行处理，然后再返回给用户。在这个示例中，模型用于处理分类问题，从推理服务返回的预测概率向量中解析出该图片的分类类别，并返回给用户。

用户可以修改这两个方法，实现自定义的 Transformer 逻辑。

### 使用 ImageBuilder 制作镜像

基于上述文件，我们编写 [Dockerfile](./Dockerfile), 然后使用 ImageBuilder 制作镜像。

为了使用 ImageBuilder，首先我们需要参照[创建 Secret](../../../build-image/build-image-on-platform/README.md#%E5%88%9B%E5%BB%BA-secret)准备上传镜像所需要的 DockerConfig `Secret`。

完成后修改 `imagebuilder.yaml` 文件，将 `spec.dockerConfig.secret` 修改为上一步中创建的 DockerConfig `Secret` 的名称，并将 `spec.tag` 修改为目标镜像，并执行以下命令：

```
kubectl apply -f imagebuilder.yaml
```

查看 `ImageBuilder` 状态，等待 Phase 一栏变为 `Succeeded`：

```sh
kubectl get -f imagebuilder.yaml -w
```

即可得到自定义的 Transformer 镜像。