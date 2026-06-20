# SAM-ATL

核心训练与推理代码目录。完整项目说明见根目录 `README.md`。

常用命令：

```bash
python train.py --config config_train.yaml
python calibrate.py --config config_calibrate.yaml
python infer.py --config config_infer.yaml
python transLearn.py --config config_trans.yaml
```

运行前请先安装根目录 `requirements.txt`，并将 SAM 预训练权重放到 `../segment-anything/checkpoints/`。
