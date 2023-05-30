[TFDV](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic) 可以用于描述统计，推断 schema，检查异常，检查数据漂移。我们使用 [Taxi Trips dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) 数据集为例。

运行

```bash
# run from parent direcotry
python3 src/tfdv/validate.py \
  --output=output/tfdv \
  --csv-data-for-inference=data/eval.csv \
  --csv-data-to-validate=data/train.csv \
  --column-names=data/column-names.json
```

TFDV 可以用于实现 rubric 中的：

* Test for Feature and Data:
  * Feature expectations are captured in a schema(1-1)
* Monitoring Tests for ML:
  * Data invariants hold in training and serving inputs(4-2)