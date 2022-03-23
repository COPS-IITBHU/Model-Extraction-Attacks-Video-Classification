# BLACKBOX

main code

## Structure

```
models/
 - modela.py
 - modelb.py
train/
 - train_loop1.py
 - train_loop2.py
generator/
 - generator1_.py
 - generator2_.py

main_1.py # usses modela with generator2 with train_loop1
..
```

## Models

- Setup

  ```shell
    #!/usr/bin/bash
    conda env create -f environment_swinT.yml
    conda activate swint
    mkdir weights/
    cd weights/
    #SwinT weights
    gdown https://drive.google.com/uc?id=10_ArqSj837hBzoQTq3RPGBZgKbBvNfSe

    #MoviNetA2 weights
    gdown --id 12jpmoKb1wfF0Mpq7IeLbeoHLtJ1YaoDV

    cd ../
  ```

- Run files from root directory of this project.

## References

ViViT - https://arxiv.org/abs/2103.15691
MoviNet - https://arxiv.org/pdf/2103.11511v2.pdf
