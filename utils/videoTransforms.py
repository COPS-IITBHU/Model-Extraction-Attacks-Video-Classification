import torchvideo.transforms as VT
from torchvision.transforms import Compose

tgan_transform = Compose([
    VT.ResizeVideo((64,64), interpolation = 2),
    VT.CollectFrames(),
    VT.PILVideoToTensor(),
    VT.NormalizeVideo(
        mean = [0.45, 0.45, 0.45],
        std = [0.5, 0.5, 0.5],
        channel_dim = 0,
        inplace = True
    )
])

transform_movinetA2 = Compose([
    VT.ToTensor(),
    VT.Resize((224, 224)),
    VT.RandomCrop((224, 224)),
])

transform_swint = Compose([
    VT.Resize(
        224, interpolation=VT.InterpolationMode.BICUBIC),
    VT.CenterCrop(224),
    # VT.RandomHorizontalFlip(p=0.5),
    # VT.RandomRotation(15),
    VT.Normalize(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
])