import math
from PIL import Image
from torch.utils.data import Dataset
import cv2
from pytorch_lightning.lite import LightningLite
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

FRAMES_PER_VIDEO = 16
NOISE_SIZE = 100
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TemporalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a sequential model to turn one vector into 16 vectors

        self.model = nn.Sequential(
            nn.ConvTranspose1d(100, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 100, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # initialize weights according to paper
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def forward(self, x):
        # reshape x so that it can have convolutions done
        x = x.view(-1, NOISE_SIZE, 1)    
        # apply the model and flip the
        x = self.model(x).transpose(1, 2)
        return x


class VideoGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # instantiate the temporal generator
        self.temp = TemporalGenerator()

        # create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # define the image generator
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # initialize weights according to the paper
        self.fast.apply(self.init_weights)
        self.slow.apply(self.init_weights)
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    def forward(self, x):
        # pass our latent vector through the temporal generator and reshape
        z_fast = self.temp(x).contiguous()
        z_fast = z_fast.view(-1, NOISE_SIZE)

        # transform the content and temporal vectors
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)
        z_slow = self.slow(x).view(-1, 256, 4, 4).unsqueeze(1)
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow]*FRAMES_PER_VIDEO,
                           dim=1).view(-1, 256, 4, 4)

        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1)

        # transform into image frames
        out = self.model(z)

        return out.view(-1, FRAMES_PER_VIDEO, 3, 64, 64).transpose(1, 2)


def singular_value_clip(w):
    dim = w.shape
    # reshape into matrix if not already MxN
    if len(dim) > 2:
        w = w.reshape(dim[0], -1)
    u, s, v = torch.svd(w, some=True)
    s[s > 1] = 1
    return (u @ torch.diag(s) @ v.t()).view(dim)


class VideoDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )

        self.conv2d = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

        # initialize weights according to paper
        self.model3d.apply(self.init_weights)
        self.init_weights(self.conv2d)

    def init_weights(self, m):
        if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, gain=2**0.5)

    def forward(self, x):
        h = self.model3d(x)
        # turn a tensor of R^NxTxCxHxW into R^NxCxHxW

        h = torch.reshape(h, (-1, 512, 4, 4))
        h = self.conv2d(h)
        return h

# dataset directary csv




# Function to extract frames
def FrameCapture(path):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    success, image = vidObj.read()
    # checks whether frames were extracted
    frems = []
    while success:

        # vidObj object calls read
        # function extract frames
        
        frems.append(image)
        # print(image)
        success, image = vidObj.read()
        count += 1

    fc = len(frems)

    frames = []
    for i in range(0, fc, (fc-1) // 15):

        image = frems[i]

        ma = max(image.shape[1], image.shape[0])
        h, w = image.shape[0], image.shape[1]

        image = cv2.copyMakeBorder(image, (ma - h) // 2, (ma-h)//2,
                                   (ma-w)//2, (ma-w)//2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        frames.append(Image.fromarray(image[:, :, [2, 1, 0]]))


    return (frames[:16], fc)


class videosDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.annotations = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

        df = pd.read_csv(csv_file)
        ll = sorted(list(df.iloc[:,1].unique()))
        self.dic = {}
        for id, i in enumerate(ll):
            self.dic[i] = id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        vid_path = os.path.join(
            self.root_dir, (self.annotations.iloc[index, 0]))
        # print(vid_path)
        vid_label = torch.tensor(self.dic[self.annotations.iloc[index, 1]])
        # put the labels into a dictionary?

        vid, fc = FrameCapture(vid_path)

        if self.transform:
            vid = self.transform(vid)

        return (vid, vid_label, fc)

import torchvideo.transforms as VT
from torchvision.transforms import Compose

transform = Compose([
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

batch_size = 32
epochs = 200
gen_lr = 3e-5
dis_lr = 3e-4
device = 'cuda'


csv_file = 'result400.csv'
root_dir = 'data'
rr = './'



from tqdm.notebook import tqdm


writer = SummaryWriter(
    f"runs/GeneratorK400"
)

dis = VideoDiscriminator()


# instantiate the generator, load the weights, and create a sample
gen = VideoGenerator()

gen.load_state_dict(torch.load('state_normal81000.ckpt')['model_state_dict'][0])


# training and sample gifs generation wrapped in pytorch lightning

class Lite(LightningLite):
    def genSamples(self, g,epochsx = 0, n = 8):
        '''
        Generate an n by n grid of videos, given a generator g
        '''
        with torch.no_grad():
            s = g(torch.rand((n**2, NOISE_SIZE))*2-1).cpu().detach().numpy()

        out = np.zeros((3, FRAMES_PER_VIDEO, 64*n, 64*n))
        for j in range(n):
            for k in range(n):
                out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n+k, :, :, :, :]

        out = out.transpose((1, 2, 3, 0))
        out = (out + 1) / 2 * 255
        out = out.astype(int)
        clip = ImageSequenceClip(list(out), fps=20)
        clip.write_gif('gifs/sample'+str(epochsx)+'.gif', fps=20)


    def run(self, epochs, gen, dis, videosDataset, gen_lr, dis_lr):
        data = videosDataset(csv_file, root_dir, transform = transform)

        train_loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
        train_loader = self.setup_dataloaders(train_loader)

        genOpt = optim.Adam(gen.parameters(), lr = gen_lr)
        disOpt = optim.Adam(dis.parameters(), lr = dis_lr)

        gen, genOpt = self.setup(gen, genOpt)
        dis, disOpt = self.setup(dis, disOpt)
        gen.train()
        dis.train()

        gen_loss_metric = []
        dis_loss_metric = []
        err = 0
        step = 0

        iteration= 0
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, 1 + epochs):
            losses = []
            for data, targets, frames in tqdm(train_loader):

                genOpt.zero_grad()
                fake = gen(torch.rand((batch_size, NOISE_SIZE))*2-1)
                pff = (dis(fake))
                gen_loss = torch.mean(-pff)
                self.backward(gen_loss)
                genOpt.step()

                disOpt.zero_grad()

                pr = dis(data)
                dis_loss = torch.mean(-pr)
                self.backward(dis_loss)
                disOpt.step()

                disOpt.zero_grad()
                fake = gen(torch.rand((batch_size, NOISE_SIZE))*2-1)
                pf = dis(fake)
                dis_loss = torch.mean(pf)
                self.backward(dis_loss)
                disOpt.step()

                gen_loss_metric.append(gen_loss)

                dis_loss_metric.append(dis_loss)

                if iteration % 5 == 0:
                    # enforcing the singular value clipping on the weights of the generator
                    for module in list(dis.children()):
                        if type(module) == nn.Conv3d or type(module) == nn.Conv2d:
                            module.weight.data = singular_value_clip(module.weight)
                        elif type(module) == nn.BatchNorm3d:
                            gamma = module.weight.data
                            std = torch.sqrt(module.running_var)
                            gamma[gamma > std] = std[gamma > std]
                            gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                            module.weight.data = gamma

            if(epoch % 5 == 0):
                self.genSamples(gen, epochsx = epoch)
                torch.save(gen.state_dict(), 'gen.pt')
                torch.save(dis.state_dict(), 'disc.pt')

            print("Epoch: " + str(epoch), flush=True)
            print(f"Genrator Loss: {sum(gen_loss_metric)/len(gen_loss_metric)}", flush = True)
            print(f"Discriminator Loss: {sum(dis_loss_metric)/len(dis_loss_metric)}", flush = True)

            writer.add_scalar(
                        "Genrator Loss", sum(gen_loss_metric)/len(gen_loss_metric), global_step=step
                    )
            writer.add_scalar(
                        "Discriminator Loss", sum(dis_loss_metric)/len(dis_loss_metric), global_step=step
                    )
            step += 1

        return (gen, dis, err, gen_loss_metric, dis_loss_metric)


print('training now!')

# gen, dis, errors, lg, ld = train_loop(epochs, gen, dis, videosDataset, gen_lr, dis_lr)

gen, dis, errors, lg, ld = Lite(strategy="ddp", devices=3, accelerator="gpu").run(epochs, gen, dis, videosDataset, gen_lr, dis_lr)
print('number of errors encountered :', errors)
print('gen-loss :', lg)
print('dis-loss :', ld)

torch.save(gen,'gen_final.pt')
torch.save(dis, 'dis_final.pt')

Lite(strategy="ddp", devices=3, accelerator="gpu").genSamples(gen)