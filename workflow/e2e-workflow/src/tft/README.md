[TFT](https://www.tensorflow.org/tfx/tutorials/transform/census) 用于预处理输入数据。例如，你可以：

* 使用均值和标准差归一化输入数据。
* 通过生成全部输入数据的词汇表，将 string 转换成 integer 。
* 通过分桶，将 float 转换成 integer 。

我们使用 [Taxi Trips dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) 数据集为例。

运行

```bash
# run in parent directory; after TFDV succeeded, as it depends on output/tfdv/schema.json
python3 src/tft/transform.py \
  --output=output/tft \
  --train=data/train.csv \
  --eval=data/eval.csv \
  --schema=output/tfdv/schema.json
```

TFT 可以用于实现 rubric 中的：

* Monitoring Tests for ML:
  * Training and serving features compute the same values(4-3)
