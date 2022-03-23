import pandas as pd
import torchvision
import torch
from vivitpytorch.vivit import *
import cv2
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
from swint_victim import SwinTransformer3D as VICTIM
from torch.utils.tensorboard import SummaryWriter


df = pd.read_csv('validation.csv')
ll = sorted(list(df.iloc[:, 1].unique()))
dic = {}
for id, i in enumerate(ll):
    dic[i] = id

transform_swint = torchvision.transforms.Compose([
    torchvision.transforms.Resize(
        224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(224),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # torchvision.transforms.RandomRotation(15),
    torchvision.transforms.Normalize(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
])


def video2img(video_path, transform=transform_swint):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    l = []
    fc = 0
    frems = []
    while success:
        frems.append(image)
        success, image = vidcap.read()
    fc = len(frems)

    if fc == 0:
        print(video_path)

    if (fc < 16):
        return torch.zeros(3, 16, 224, 224)
    
    for i in range(0, fc, (fc-1)//15):
        image = frems[i]
        l.append(
            transform(
                torch.tensor(image).type(
                    torch.FloatTensor).permute(2, 0, 1)
            )
        )
    if len(l) == 0:
        return torch.zeros(3, 16, 224, 224)
    else:
        return torch.stack(l[:16], dim=1)


class videosDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=transform_swint):

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        vid_path = os.path.join(
            self.root_dir, (self.annotations.iloc[index, 0]))
        vid_label = torch.tensor(dic[self.annotations.iloc[index, 1]])
        vid = video2img(vid_path, self.transform)
        return (vid, vid_label)


batch_size = 16
epochs = 2000
device = 'cuda:0'
csv_file = "result.csv"
root_dir = "/"
rr = root_dir
train_data = videosDataset(csv_file, rr, transform=transform_swint)
test_data = videosDataset("test.csv", rr, transform=transform_swint)

# test_data = videosDataset(
#     "validation.csv", rr, transform=transform_swint)


writer = SummaryWriter("runs_GB")

train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                          shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(
    dataset=test_data, batch_size=batch_size, pin_memory=True, num_workers=4)
os.listdir(rr)

losses = []

teacher = VICTIM()
teacher.load_state_dict(torch.load('swint_victim_pretrained.pth', map_location="cpu"))
teacher.eval()
teacher.to(device, non_blocking=True)


model = ViViT(224, 16, 400, 16).to(device, non_blocking=True)
model.load_state_dict(torch.load('weights/vivit_swint.pth', map_location="cpu"))
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
        out = model(img)
        del img
        # hello bye
        with torch.no_grad():
            target = teacher(image)
            pred = target.argmax(dim=1)
        del image
        correct_victim += pred.eq(label.view_as(pred)).sum().item()
        loss = loss_func(out, pred)
        del out, pred, label 
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
        for data, target in test_loader:
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
        torch.save(model.state_dict(), 'weights/vivit_swint.pth')
        best_acc = accuracy
    
print('loss till epoch=', epoch, sum(losses)/(len(losses)))

# # Validation Test
# correct = 0
# test_loss = 0
# with torch.no_grad():
#     for data, target in tqdm(test_loader):
#         data, target = data.to(device), target.to(device)
#         output = model(data.permute(0, 2, 1, 3, 4))
#         pred = output.argmax(dim=1)
#         test_loss += F.cross_entropy(output,target, reduction='sum').item()
#         correct += pred.eq(target.view_as(pred)).sum().item()
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct,
#           len(test_loader.dataset), accuracy))
