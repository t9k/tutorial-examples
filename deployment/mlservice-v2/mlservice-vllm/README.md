# 基于 MLService 部署大模型

本示例使用 MLService 部署一个大模型，其中部署框架使用 [vllm](https://docs.vllm.ai)，部署的模型使用 [通义千问-1.8B](https://huggingface.co/Qwen/Qwen-1_8B-Chat)

## 准备模型

本示例使用的模型为[通义千问-1.8B](https://huggingface.co/Qwen/Qwen-1_8B-Chat)。为了获得更快的下载速度，我们使用国内模型平台 [ModelScope](https://modelscope.cn/) 来下载此模型。

安装 `modelscope` 依赖，并设置环境变量 `MODELSCOPE_CACHE`，该环境变量指定了下载模型的路径。然后执行 `download.py` 文件下载模型：

```
# cd into current directory
cd ~/tutorial-examples/deployment/mlservice-v2/mlservice-vllm
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
export MODELSCOPE_CACHE=/t9k/mnt/modelscope
python download.py
```

## 部署模型

创建 MLServiceRuntime：

```bash
kubectl create -f runtime.yaml
```

执行以下命令来部署服务：

``` bash
kubectl apply -f ./mlservice.yaml
```

在命令行监控服务是否准备好：

``` bash
kubectl get -f ./mlservice.yaml -w
```

等待 `Ready` 一栏变为 `True`，便可开始使用服务。

## 使用服务

``` bash
address=$(kubectl get -f ./mlservice.yaml -ojsonpath='{.status.address.url}')
curl ${address}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "prompt": "Once upon a time, there was",
    "max_tokens": 50,
    "temperature": 0.5
  }'
```

你会得到类似下面的输出：

```
{
  "id": <id>,
  "object":"text_completion",
  "created":<time>,
  "model":"qwen",
  "choices":[
    {
      "index":0,
      "text":" a small village nestled in the heart of a dense forest. The villagers were simple people who lived in harmony with nature, and they were proud of their way of life. They spent their days working in the fields and tending to their livestock, and",
      "logprobs":null,
      "finish_reason":"length"
    }],
    "usage":{
      "prompt_tokens":7,
      "total_tokens":57,
      "completion_tokens":50}
}
```