# 平台教程示例

`t9k/tutorial-examples` 仓库存放了 TensorStack AI 计算平台的所有教程示例。

## 使用方法

1. 在您的项目中创建一个名为 tutorial、大小 4 Gi 以上的 PVC，然后创建一个同样名为 tutorial 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

2. 进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库。

```shell
cd ~
git clone https://github.com/t9k/tutorial-examples.git
```

3. 继续使用 **Notebook 的终端**，参照各示例的 README 进行操作，或参照各示例在用户文档中对应的教程进行操作。各示例的 README 与相应教程中给出的操作步骤是一致的。
