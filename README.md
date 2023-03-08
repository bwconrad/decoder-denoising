# Decoder Denoising Pretraining for Semantic Segmentation

PyTorch reimplementation of ["Decoder Denoising Pretraining for Semantic Segmentation"](https://arxiv.org/abs/2205.11423).

<p align="center">
<img src="assets/figure.png" width="50%" style={text-align: center;}/>
</p>

## Requirements
- Python 3.8+
- `pip install -r requirements`

## Usage
To perform decoder denoising pretraining on a U-Net with a ResNet-50 encoder run:
```
python train.py --gpus 1 --max_epochs 100 --data.root path/to/data/ --model.arch unet --model.encoder resnet50 
```

- `--model.arch` can be one of `unet, unetplusplus, manet, linknet, fpn, pspnet, deeplabv3, deeplabv3plus, pan`.
- `--model.encoder` can be any from the list [here](https://smp.readthedocs.io/en/latest/encoders.html).
- `configs/` contains example configuration files which can be run with `python train.py --config path/to/config`.
- Run `python train.py --help` to get descriptions for all the options.

### Using a Pretrained Model
Model weights can be extracted from a pretraining checkpoint file by running:
```
python scripts/extract_model_weights.py -c path/to/checkpoint/file
```
You can then initialize a segmentation model with these weights with the following (example for U-Net with ResNet-50 encoder):
```python
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

weights = torch.load("weights.pt")

model = smp.create_model(
    "unet",
    encoder_name="resnet50",
    in_channels=3,
    classes=3, # Same number used during pretraining for now
    encoder_weights=None,
)

model.load_state_dict(weights, strict=True)

# Replace segmentation head for fine-tuning
in_channels = model.segmentation_head[0].in_channels
num_classes = 10
model.segmentation_head[0] = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
```

## Citation
```bibtex
@inproceedings{brempong2022denoising,
  title={Denoising Pretraining for Semantic Segmentation},
  author={Brempong, Emmanuel Asiedu and Kornblith, Simon and Chen, Ting and Parmar, Niki and Minderer, Matthias and Norouzi, Mohammad},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4175--4186},
  year={2022}
}
```
