# SICBSeg: Collapsed Building Segmentation


## Project Structure

```text
SAM-ATL/                 main body
segment-anything/        The official SAM Python package
segment-anything/checkpoints/     Placing SAM pre-training weights
```

## Setup

It is recommended to use Python 3.10. Before GPU training, install the matching PyTorch according to the native CUDA version.

```bash
pip install -r requirements.txt
```


Download SAM weights to `segment-anything/checkpoints/`

```text
segment-anything/checkpoints/sam_vit_l_0b3195.pth
```

## Data

Placing data in the following structure `data/` :

```text
SAM-ATL/data/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
```

## Usage

```bash
cd SAM-ATL
python train.py --config config_train.yaml
python calibrate.py --config config_calibrate.yaml
python infer.py --config config_infer.yaml
```

Transfer training:

```bash
python transLearn.py --config config_trans.yaml
```



