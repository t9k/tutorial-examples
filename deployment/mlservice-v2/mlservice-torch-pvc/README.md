# 部署用于生产环境的模型推理服务（PyTorch 模型）

本示例使用 MLService 部署用于生产环境的模型推理服务（以 PyTorch 模型为例），主要包含以下操作：

1. 创建 `MLServiceRuntime` 用于部署 PyTorch 类型的模型
2. 创建 `MLService` 模型推理服务，其使用 PVC 中存储的模型
3. 使用命令行向 `MLService` 发送推理请求

关于如何将 PyTorch 自定义训练模型打包为本示例所使用的 torch model archive（.mar）文件，请参阅官方示例 [Digit recognition model with MNIST dataset](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)。

## 操作步骤

切换到当前目录下，使用 `runtime.yaml` 创建 MLServiceRuntime：

```shell
# cd into current directory
cd ~/tutorial-examples/deployment/mlservice-v2/mlservice-torch-pvc
kubectl create -f runtime.yaml
```

使用 `mlservice.yaml` 创建 MLService：

```shell
kubectl create -f mlservice.yaml
```

在命令行监控服务是否准备好：

```shell
kubectl get -f mlservice.yaml -w
```

待其 `READY` 值变为 `true` 后，通过发送如下请求测试该推理服务：

```shell
address=$(kubectl get -f mlservice.yaml -ojsonpath='{.status.address.url}')
curl -T test_data/0.png ${address}/v1/models/mnist:predict # or use `1.png`, `2.png`
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
