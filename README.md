# Model Extraction Attacks Video Classification

## Installation

```bash
$ pip install vidmodex ## Stable version
$ pip install git+https://github.com/hex-plex/Model-Extraction-Attacks-Video-Classification ## Latest development
```

## Usage
Simple snippet explaining the usage

### BlackBox Extraction
```python
# Black Box Victim: SwinT, Student: ViViT, Generator: Tgan

from vidmodex.models import ViViT as Student
from vidmodex.models import SwinT as Victim
from vidmodex.generator import Tgan as Generator

custom_config = {}
custom_config["num_classes"] = 400

blackbox_main(custom_config)
```
### GreyBox Extraction

```python
# Grey Box Victim: SwinT, Student: ViViT, Generator: Tgan, Dataset: Kinetics 400

from vidmodex.models import ViViT as Student
from vidmodex.models import SwinT as Victim
from vidmodex.generator import Tgan as Generator

custom_config = {}
custom_config["csv_file"] = "ENTER-THE-LOCATION-OF-DATA-CSV"
custom_config["root_dir"] = "ENTER-THE-LOCATION-OF-DATA-ROOT"
custom_config["ucf_gan_weights"] = "ENTER-THE-LOCATION-OF-UCF-WEIGHTS" or "state_normal81000.ckpt"
custom_config["num_classes"] = 400

greybox_main(custom_config)
```

## File Structure

This is for reference if one wants to experiment his own model or algorithm he may change that specific module / part

```
models/
 - modela.py           ## Video Classification Architecture (Teacher/Student)
 - modelb.py
train/
 - train_loop1.py      ## Traing Algorithm
 - train_loop2.py
generator/
 - generator1_.py      ## Video Generator Architecture
 - generator2_.py

main_file.py           ## Contains your custom config/data
..
```

## References

ViViT - https://arxiv.org/abs/2103.15691
MoviNet - https://arxiv.org/pdf/2103.11511v2.pdf
