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