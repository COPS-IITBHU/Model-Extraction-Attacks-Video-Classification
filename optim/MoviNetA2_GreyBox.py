# -*- coding: utf-8 -*-
import pandas as pd
import torchvision
import torch
from vivitpytorch.vivit import *
import cv2
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from movinets.models import MoViNet
from movinets.config import _C

transform_movinetA2 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomCrop((224, 224)),
])

df = pd.read_csv(
    'result600.csv')
ll = sorted(list(df.iloc[:, 0].unique()))
dic = {}
for id, i in enumerate(ll):
    dic[i] = id


def video2img(video_path, transform=transform_movinetA2):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    l = []
    fc = 0
    frems = []
    while success:
        frems.append(image)
        success, image = vidcap.read()
    fc = len(frems)
    for i in range(0, fc, (fc - 1) // 15):
        image = frems[i]
        # print(image.shape)
        l.append(
            transform(image)
        )
    if len(l) == 0:
      print(video_path)
      return torch.zeros(3, 16, 224, 224)
      
    else:
      return torch.stack(l[:16], dim=1)

class videosDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=transform_movinetA2):

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        vid_path = os.path.join(
            self.root_dir, self.annotations.iloc[index, 0][81:])
        vid_label = torch.tensor(dic[self.annotations.iloc[index, 1]])
        vid = video2img(vid_path, self.transform)
        return (vid, vid_label)

batch_size = 24
epochs = 2000
device = 'cuda:0'
csv_file = "result_k600f.csv"
root_dir = "/DATA/shorya/experiment/datafree-model-extraction/dfme/data/kinetics600/"
rr = root_dir
train_data = videosDataset(csv_file, rr, transform=transform_movinetA2)
test_data = videosDataset("test600.csv", rr, transform=transform_movinetA2)
writer = SummaryWriter("runs_GB_A2")

train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                          shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(
    dataset=test_data, batch_size=batch_size, pin_memory=True, num_workers=4)
# os.listdir(rr)

losses = []

teacher =  MoViNet(_C.MODEL.MoViNetA2, causal = False, num_classes = 600, tf_like = True)
teacher.load_state_dict(torch.load("./modela2"))
teacher.eval()
teacher.to(device, non_blocking=True)

model = ViViT(224, 16, 600, 16).to(device, non_blocking=True)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(
    0.9, 0.999), eps=1e-8, weight_decay=0.02,)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)

step = -1
best_acc = 0
batches_per_epoch = 100
for epoch in tqdm(range(epochs)):
    step += 1
    itr = 0 
    
    # Train
    losses2 = []
    correct_victim  = 0
    for image, label in tqdm(train_loader):
        if (itr == batches_per_epoch):
            break
        itr += 1
        label = label.cuda()
        image = image.to(device, non_blocking=True)
        img = image.permute(0, 2, 1, 3, 4)
        img2 = image
        out = model(img)
        # del img
        with torch.no_grad():
            target = teacher(img2)
            pred = target.argmax(dim=1)
        del image, img2
        correct_victim += pred.eq(label.view_as(pred)).sum().item()
        loss = loss_func(out, pred)
        del pred, label 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        losses2.append(loss)
    lr_scheduler.step()
    print('Victim accuracy : {}'.format(100*correct_victim/batches_per_epoch/batch_size))
    writer.add_scalar("Victim accuracy ",(100*correct_victim/batches_per_epoch/batch_size), global_step=step)
    
    # Test
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data.permute(0, 2, 1, 3, 4))
            pred = output.argmax(dim=1)
            test_loss += F.cross_entropy(output,target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct,
              len(test_loader.dataset), accuracy))
       
        print('loss at epoch =', epoch, sum(losses2)/(len(losses2)))


    writer.add_scalar("Loss against victim", sum(
        losses2)/(len(losses2)), global_step=step)
    writer.add_scalar("Test Loss", test_loss, global_step=step)
    writer.add_scalar("Test Acc", accuracy, global_step=step)
    if(accuracy > best_acc):
        torch.save(model.state_dict(), 'weights/vivit_A2.pth')
        best_acc = accuracy
    
print('loss till epoch=', epoch, sum(losses)/(len(losses)))
