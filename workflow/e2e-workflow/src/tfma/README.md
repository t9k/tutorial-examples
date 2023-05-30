# TL;DR

使用 [TFMA](https://www.tensorflow.org/tfx/tutorials/model_analysis/tfma_basic) 分析 [trainer](../trainer) 训练好的模型。

## 运行

Install deps

```bash
pip3 install -r requirements.txt
````

```bash
python3 \
  src/tfma/model_analysis.py \
  --output=output/tfma \
  --model=output/trainer/model/tfma_eval_model_dir \
  --eval=data/eval.csv \
  --schema=output/tfdv/schema.json \
  --slice-columns=trip_start_hour
```

可以用于实现 rubric 中的：

* Test for Model Development:
  * Model quality is sufficient on all important data slices(2-6)
* Test for ML Infrastructure:
  * Model quality is validated before attempting to serve it(3-4)
