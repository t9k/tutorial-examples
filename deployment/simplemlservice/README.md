# SimpleMLService Example

使用 SimpleMLService 部署推理服务的示例。模型信息如下：
* 功能：图片分类。
* 框架：tensorflow
* 数据集：keras.datasets.fashion_mnist


## 解压模型

切换到本文档所在的目录下，解压 saved_model.tar.gz 文件：

```bash
# cd into current directory
cd ~/tutorial-examples/deployment/simplemlservice

tar zxvf saved_model.tar.gz
```

得到的 `saved_model` 目录里面是一个在 MNIST 数据集上训练的简易 Keras 模型以 SavedModel 格式保存的文件。


## 部署服务

在当前目录下，使用 [simplemlservice.yaml](./simplemlservice.yaml) 创建 SimpleMLService：

```sh
$ kubectl apply -f ./simplemlservice.yaml
```

## 查看 SimpleMLService 状态

运行下列命令查看 SimpleMLService、Pod、Service 的状态：

```sh
$ kubectl get simplemlservice mnist
NAME    READY   URL                                                             AGE
mnist   True    managed-simplemlservice-mnist-947d0.demo.svc.cluster.local   22s
$ kubectl get pod -l simplemlservice=mnist
NAME                                                   READY   STATUS    RESTARTS   AGE
managed-simplemlservice-mnist-947d0-647477b8f8-p6pvn   1/1     Running   0          37s
$kubectl get svc 
NAME                                 TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
managed-simplemlservice-mnist-947d0  ClusterIP      10.233.62.14    <none>        80/TCP     46s
```

## 测试服务

当 Pod 处于 Running 状态之后，可以向 SimpleMLService 服务发送预测请求，步骤如下:

首先查看 `simplemlservice` 在集群内的 URL：

```sh
$ kubectl get simplemlservice mnist
NAME    READY   URL                                                             AGE
mnist   True    managed-simplemlservice-mnist-947d0.demo.svc.cluster.local   22s
```

然后发送预测请求：

```sh
$ curl -d @./data.json http://managed-simplemlservice-mnist-947d0.demo.svc.cluster.local/v1/models/mnist:predict
{
    "predictions": [[-6.4451623, -8.87918, -4.54794025, -6.75463295, -5.6717906, 0.776205719, -4.46030855, 2.78089452, 0.137464285, 5.99647713]
    ]
}
```

预测结果：data.json 所对应的图片是数字 0～9 的概率。
