import cv2
import requests
import torch
from torchvision import transforms
from tqdm import tqdm
import os
import sys
sys.path.insert(1, './')
from models.swint_victim import SwinTransformer3D as VICTIM
from models.swint_student import SwinTransformer3D as STUDENT

USE_CUDA = True
PRETRAINED = True
TRAIN = True

VIDEO_PATH = 'train/demo.mp4'  # TODO : to be changed based upon the combined training loop
LABEL = 6  # TODO : to be changed based upon the combined training loop

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
])


def video2img(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    l = []
    while success:
        if count % 20 == 0:
            l.append(
                transform(
                    torch.tensor(image).type(
                        torch.FloatTensor).permute(2, 0, 1)
                ).unsqueeze(dim=0)
            )
        success, image = vidcap.read()
        count += 1
    return torch.stack(l, dim=2)


def use_pretrained(model,
                   folder='weights/',
                   file_name="swint_victim_pretrained.pth",
                   download=False,
                   url=None, ):
    if download:
        response = requests.get(url, stream=True)
        t = int(response.headers.get('content-length', 0))  # total file size
        block_size = 1024 ** 2  # 1 Mbit
        progress_bar = tqdm(total=t, unit='iB', unit_scale=True)
        with open(f"weights/{file_name}", 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if (t != 0) and (progress_bar.n != t):
            print("ERROR downloading weights!")
            return -1
        print(f"Weights downloaded in {folder} directory!")
    model.load_state_dict(torch.load(os.path.join(folder, file_name)))
    return model


student_model = STUDENT()
victim_model = VICTIM()
if PRETRAINED:
    victim_model = use_pretrained(victim_model)
if USE_CUDA:
    student_model.cuda()
    victim_model.cuda()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.02
)
victim_model.eval()
while TRAIN:
    """
    This will run until query is being generated by the generator or else generator \
    will change TRAIN to False.
    """
    # Converting video to image of size [BatchSize=1, channels=3, frames, height=224, width=224]
    image = video2img(VIDEO_PATH)
    if USE_CUDA:
        image = image.cuda()
    # Querying the label from victim model
    with torch.no_grad():
        LABEL = torch.argmax(victim_model(image), 1).item()
    target = torch.zeros(1, 400)  # (BatchSize, Classes)
    target[:, LABEL] = 1
    if USE_CUDA:
        target = target.cuda()
    # Training the student model
    output = student_model(image)
    loss = loss_func(output, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()