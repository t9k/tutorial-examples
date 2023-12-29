# SimpleMLService Example

使用 SimpleMLService 部署推理服务的示例。模型信息如下：
* 模型功能：图片分类。
* 模型框架：tensorflow
* 数据集：keras.datasets.fashion_mnist

## 部署服务

切换到当前目录下，使用 [simplemlservice.yaml](./simplemlservice.yaml) 创建 SimpleMLService：
```bash
$ kubectl apply -f ./simplemlservice.yaml
```

## 查看 SimpleMLService 状态

运行下列命令查看 SimpleMLService、Pod、Service 的状态：

```bash
$ kubectl get simplemlservice
NAME     DNS
mnist    mnist.czx.svc.cluster.local
$ kubectl get pod -l deployment=managed-simplemlservice-mnist
NAME                                             READY   STATUS    RESTARTS   AGE
managed-simplemlservice-d6fad-7499fccdfc-scx42   1/1     Running   0          5m
$ kubectl get svc
NAME          TYPE           CLUSTER-IP      EXTERNAL-IP        PORT(S)   AGE
mnist         ClusterIP      10.233.19.65    <none>             80/TCP    5m57s
```

## 测试服务

当 Pod 处于 Running 状态之后，可以向 SimpleMLService 服务发送预测请求，步骤如下。

首先将 service mnist 的 80 端口转发到本地的 8080 端口：
```bash
$ kubectl port-forward svc/mnist 8080:80
```

然后发送预测请求：
```bash
$ curl -d @./data.json localhost:8080/v1/models/mnist:predict
{
    "predictions": [[-6.4451623, -8.87918, -4.54794025, -6.75463295, -5.6717906, 0.776205719, -4.46030855, 2.78089452, 0.137464285, 5.99647713]
    ]
}
```

查看实际结果，可以发现与预期结果一致：
```bash
$ cat result.json 
{"labels": [9]}
```
