训练一个 wide & deep 分类器。

运行

```bash
# run in parent dir; make sure TFT succeeded
python3 \
  src/trainer/task.py \
  --output=output/trainer \
  --transformed-data-dir=output/tft \
  --predict-data=data/predict.csv \
  --target=tips \
  --learning-rate=0.01 \
  --hidden-layer-size=1500 \
  --steps=3000 \
  --epochs=1
```

可以用于实现 rubric 中的：

* Test for Feature and Data: 
  * Features adhere to meta-level requirements(1-4)
  * New feature can be added quickly(1-6)
* Test for ML Infrastructure:
  * Training is reproducible(3-1)