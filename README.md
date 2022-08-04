# Decoder Denoising Pretraining for Semantic Segmentation

PyTorch reimplementation of ["Decoder Denoising Pretraining for Semantic Segmentation"](https://arxiv.org/abs/2205.11423).

<p align="center">
<img src="assets/figure.png" width="50%" style={text-align: center;}/>
</p>

### Requirements
- Python 3.8+
- `pip install -r requirements`

### Usage
To decoder pretrain a U-Net with a ResNet-50 encoder using the default settings run:
```
python train.py --gpus 1 --max_epochs 100 --data.root path/to/data/ --model.arch unet --model.encoder resnet50 
```

- `--model.arch` can be one of `unet, unetplusplus, manet, linknet, fpn, pspnet, deeplabv3, deeplabv3plus, pan`.
- `--model.encoder` can be any from the list [here](https://github.com/qubvel/segmentation_models.pytorch#encoders).
- `configs/` contains example configuration files which can be run with `python train.py --config path/to/config`.
- Run `python train.py --help` for a list and description for all options.

