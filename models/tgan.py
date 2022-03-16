import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import Image
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
# !pip install opencv-python-headless
FRAMES_PER_VIDEO = 16
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def genSamples(g,epochsx = 0, n = 8):
    '''
    Generate an n by n grid of videos, given a generator g
    '''
    with torch.no_grad():
        s = g(torch.rand((n**2, 2000), device='cuda')*2-1).cpu().detach().numpy()

    out = np.zeros((3, FRAMES_PER_VIDEO, 64*n, 64*n))
    # print(out.shape)
    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n+k, :, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    out = out.astype(int)
    clip = ImageSequenceClip(list(out), fps=20)
    clip.write_gif('gifs/sample'+str(epochsx)+'.gif', fps=20)


# ## How to Generate Videos
# The first thing to note about video generation is that we are now generating tensors with an added dimension. While conventional image methods work to generate tensors in $\mathbb{R}^{C \times H \times W}$, we are now generating tensors of size $\mathbb{R}^{T \times C \times H \times W}$.
# 
# To solve this problem, TGAN proposed generating temporal dynamics first, then generating images. Gordon and Parde, 2020 have a visual that summarizes the generator's process.
# 
# ![generator](https://imgur.com/vH8cakL.png)
# 
# A latent vector $\vec{z}_c$ is sampled from a distribution. This vector is fed into some generic $G_t$ and it transforms the vector into a series of latent temporal vectors. $G_t:\vec{z}_c \mapsto \{\vec{z}_0, \vec{z}_1, \dots, \vec{z}_t\}$ From there each temporal vector is joined with $\vec{z}_c$ and fed into an image generator $G_i$. With all images created, our last step is to concatenate all of the images to form a video. Under this setup we decompose time and the images.
# 
# Today we will be trying to represent the UCF101 dataset. This dataset is composed of 101 action classes. Below is a sample of real examples:
# 
# ![gif grid](https://imgur.com/9Cp5868.gif)
# 
# ## The Temporal Generator $G_t$
# Here we will be implementing our temporal generator. It transforms a vector in $\mathbb{R}^{100}$ to multiple (16 to be exact) vectors in $\mathbb{R}^{100}$. In TGAN they used a series of transposed 1D convolutions, we will discuss the limitations of this choice later. 



class TemporalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a sequential model to turn one vector into 16
        
                # (2000, 1)
        
        self.model = nn.Sequential(
            
            nn.ConvTranspose1d(2000, 4096, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.ConvTranspose1d(4096, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.ConvTranspose1d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.ConvTranspose1d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.ConvTranspose1d(1024, 2000, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            
        )
        

        
        # self.model = nn.Sequential(
        #     nn.ConvTranspose1d(100, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(128, 100, kernel_size=4, stride=2, padding=1),
        #     nn.Tanh()
        # )

        # initialize weights according to paper
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def forward(self, x):
        # reshape x so that it can have convolutions done 
        x = x.view(-1, 2000, 1)     # 100 -> 2000
        # apply the model and flip the 
        x = self.model(x).transpose(1, 2)
        return x


# ## Putting It All Together
# With our $\vec{z}_c$ generated, and our temporal vectors created, it is time to generate our individual images. The first step is to map the two vectors into appropriate sizes to be fed into a transposed 2D convolutional kernel. This is done by a linear transformation with a nonlinearity. Each newly transformed vector is reshaped to a tensor of $\mathbb{R}^{256 \times 4 \times 4}$. In this shape the two sets of vectors are concatenated across the channel dimension.
# After the vectors are transformed, reshaped, and concatenated, it's finally time for us to make the images! TGAN ensues with a generic image generator of multiple transposed 2D convolutions. After enough transposed convolutions, batchnorms, and ReLUs, the final two operations are a transposed convolution to 3 color channels and a $\tanh$ activation. Our last step is to alter the shape so that the tensor has time, color-channel, height, and width dimensions. We now have a video!

class VideoGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # instantiate the temporal generator
        self.temp = TemporalGenerator()

        # create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(2000, 1024 * 4**2, bias=False), # 100 -> 2000
            nn.BatchNorm1d(1024 * 4**2),             # 256 - 4096
            nn.ReLU(),
            nn.Linear(1024 * 4**2, 512 * 4**2, bias=False), # 100 -> 2000
            nn.BatchNorm1d(512 * 4**2),             # 256 - 4096
            nn.ReLU(),
            nn.Linear(512 * 4**2, 256 * 4**2, bias=False), # 100 -> 2000
            nn.BatchNorm1d(256 * 4**2),             # 256 - 4096
            nn.ReLU()
        )

        # create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(2000, 1024 * 4**2, bias=False), # 100 -> 2000
            nn.BatchNorm1d(1024 * 4**2),             # 256 - 1024
            nn.ReLU(),
            nn.Linear(1024 * 4**2, 512 * 4**2, bias=False), # 100 -> 2000
            nn.BatchNorm1d(512 * 4**2),             # 256 - 4096
            nn.ReLU(),
            nn.Linear(512 * 4**2, 256 * 4**2, bias=False), # 100 -> 2000
            nn.BatchNorm1d(256 * 4**2),             # 256 - 4096
            nn.ReLU()
        )

        # define the image generator
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
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
        z_fast = z_fast.view(-1, 2000)

        # transform the content and temporal vectors 
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)
        z_slow = self.slow(x).view(-1, 256, 4, 4).unsqueeze(1)
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow]*FRAMES_PER_VIDEO, dim=1).view(-1, 256, 4, 4)

        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1)

        # transform into image frames
        out = self.model(z)

        return out.view(-1, FRAMES_PER_VIDEO, 3, 64, 64).transpose(1, 2)


# def split(a, n):
#     k, m = divmod(len(a), n)
#     return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))def split(a, n):
#     k, m = divmod(len(a), n)
#     return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))### The Discriminator
# 
# We're no longer operating on images, so now we need to rethink our discriminator. 2D convolutions won't work due to our time dimension, what should we do? TGAN proposes a discriminator composed of a series of 3D convolutions and singular 2D convolution. From one video it produces a single integer.
# class VideoDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model3d = nn.Sequential(
#             nn.Conv3d(3, 64, kernel_size=4, padding=1, stride=2),
#             nn.LeakyReLU(0.2),
#             nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2),
#             nn.BatchNorm3d(512),
#             nn.LeakyReLU(0.2)
#         )
# 
#         self.conv2d = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
# 
#         # initialize weights according to paper
#         self.model3d.apply(self.init_weights)
#         self.init_weights(self.conv2d)
# 
#     def init_weights(self, m):
#         if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
#             nn.init.xavier_normal_(m.weight, gain=2**0.5)
# 
#     def forward(self, x):
#         h = self.model3d(x)
#         # turn a tensor of R^NxTxCxHxW into R^NxCxHxW
#         h = torch.reshape(h, (32, 512, 4, 4))
#         h = self.conv2d(h)
#         return h
# ```python
# class VideoDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model3d = nn.Sequential(
#             nn.Conv3d(3, 64, kernel_size=4, padding=1, stride=2),
#             nn.LeakyReLU(0.2),
#             nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2),
#             nn.BatchNorm3d(512),
#             nn.LeakyReLU(0.2)
#         )
# 
#         self.conv2d = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
# 
#         # initialize weights according to paper
#         self.model3d.apply(self.init_weights)
#         self.init_weights(self.conv2d)
# 
#     def init_weights(self, m):
#         if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
#             nn.init.xavier_normal_(m.weight, gain=2**0.5)
# 
#     def forward(self, x):
#         h = self.model3d(x)
#         # turn a tensor of R^NxTxCxHxW into R^NxCxHxW
#         h = torch.reshape(h, (32, 512, 4, 4))
#         h = self.conv2d(h)
#         return h
# ```
# Once our discriminator performs inference on some samples the generated integers are then used in the WGAN formulation (you'll learn more about this next week!):
# 
# $$\operatorname*{argmax}_D \operatorname*{argmin}_G\mathbb{E}_{x\sim \mathbb{P}_r}[D(x)]-\mathbb{E}_{z\sim p(z)}[D(G(z))]$$
# 
# During training this looks like the following.
# 
# ```python
# # update discriminator
# pr = dis(real)
# fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
# pf = dis(fake)
# dis_loss = torch.mean(-pr) + torch.mean(pf)
# dis_loss.backward()
# disOpt.step()
# 
# # update generator
# genOpt.zero_grad()
# fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
# pf = dis(fake)
# gen_loss = torch.mean(-pf)
# gen_loss.backward()
# genOpt.step()
# ```
# 

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
        # print('x-before :', x.shape)
        sh = x.shape
        if(sh[1] == 16):
            x = x.permute(0, 2, 1, 3, 4)
        # print('x in c3d:',x.shape)
        # print(x.type)
        h = self.model3d(x)
        # print(h.shape)
        # print(h.type)
        # turn a tensor of R^NxTxCxHxW into R^NxCxHxW
        if(h.shape[-1]==22):
            h = torch.reshape(h, (-1, 512, 22, 22))
        else:
            h = torch.reshape(h, (-1, 512, 4, 4))
        h = self.conv2d(h)
        return h









dis = VideoDiscriminator().cuda()


# This model took 16 hours to train on an RTX-2080ti, so we'll use a pretrained version to explore the results.
# 
# Note: Make sure to use a GPU runtime!



# instantiate the generator, load the weights, and create a sample
gen = VideoGenerator().cuda()

# gen.load_state_dict(torch.load('state_normal81000.ckpt')['model_state_dict'][0])

gen = torch.load('gen5.pt')
dis = torch.load('disc4.pt')

# genSamples(gen)
# def display_gif(fn):
#     from IPython import display
#     return display.HTML('<img src="{}">'.format(fn))
# display_gif("sample.gif")




import pandas as pd
df = pd.read_csv('result400.csv')
# ll = df.iloc[:,1]


# df = pd.read_csv('/home/ug2019/eee/19085023/datafree-model-extraction/dfme/result400.csv')
ll = sorted(list(df.iloc[:,1].unique()))
# dic = {}
# for id, i in enumerate(ll):
#     dic[i] = id


# ll.tolist()
# ll = set(ll)
dic = {}
for id, i in enumerate(ll):
    dic[i] = id
# dic





import pandas as pd
import cv2
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

# Function to extract frames
def FrameCapture(path):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    success, image = vidObj.read()
    # checks whether frames were extracted
    # success = 1
    frems = []
    while success:

        # vidObj object calls read
        # function extract frames
        

        # Saves the frames with frame-count
        # print(image.shape)
        # print(image)
        
        
        frems.append(image)
        # print(image)
        success, image = vidObj.read()
        count += 1
        
    fc = len(frems) 
    
    # print(fc)
    
    # fc = count
    # print(count)
    frames = []
    # indices = arange()
    for i in range(0, fc, (fc-1)//15):
        
        image = frems[i]
        
        image = cv2.resize(image, dsize=(360, 360), interpolation=cv2.INTER_AREA)
        
        frames.append(image)
    
    npa = np.asarray(frames)
    
    ft = torch.FloatTensor(npa)
    
    # RuntimeError: Given groups=1, weight of size [64, 3, 4, 4, 4], expected input[1, 17, 360, 480, 3] to have 3 channels, but got 17 channels instead
    x = ft[:16].shape
    # print('x shape =', x)
    ft = ft[:16].reshape(x[0], 3, 360, 360)
    
    return (ft, fc)


class videosDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform = None):
        
        self.annotations = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    
    def __getitem__(self, index):
        vid_path = os.path.join(self.root_dir, (self.annotations.iloc[index, 0]))
        # print(vid_path)
        vid_label = torch.tensor(dic[self.annotations.iloc[index, 1]])
        # put the labels into a dictionary?
        
        vid, fc = FrameCapture(vid_path)
        
        if self.transform:
            vid = self.transform(vid)
        
        return (vid, vid_label, fc)
        
        





# pip install opencv-python-headless

# path = '217592rSS_c_000004_000014.mp4'

# x = FrameCapture(path)

# ([17, 360, 360, 3])

# pwd

batch_size = 16
epochs = 50
gen_lr = 2e-5
dis_lr = 2e-5
device = 'cuda'
# genSamples(gen, epochsx = 1 + epochs)

# frames per video = 16
# disOpt = 
# genOpt = 

csv_file = 'result400.csv'
root_dir = '/tmp'
rr = './'
data = videosDataset(csv_file, rr, transform = None)

train_loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)





l = [10, 20, 30]





sum(l)/len(l)





len(train_loader)





# from tqdm.notebook import tqdm


writer = SummaryWriter(
    f"runs/GeneratorK400"
)


def train_loop(epochs, gen, dis, videosDataset, gen_lr, dis_lr):
    # criterion = nn.CrossEntropyLoss()
    genOpt = optim.Adam(gen.parameters(), lr = gen_lr)
    disOpt = optim.Adam(dis.parameters(), lr = dis_lr)
    gen_loss_metric = []
    dis_loss_metric = []
    err = 0
    step = 0

    iteration= 0
    for epoch in tqdm(range(epochs)):
        losses = []
        # pbar = tqdm(range(len(train_loader), desc="Batch:")
        for data, targets, frames in tqdm(train_loader):
            

            # print(type(real))
            # out_g = gen()
            # update discriminator
            # "real" sequence of imges comes from the dataset
            #  torch.Size([32, 3, 16, 64, 64]) <class 'torch.Tensor'>
            # pr = torch.zeros([32, 3, 16, 64, 64])
            # try:

                
                            

            data = data.to(device)
            targets = targets.to(device)
            pr = dis(data)
            # print("step 1")

            fake = gen(torch.rand((batch_size, 2000), device='cuda')*2-1)
            # print(fake.shape, type(fake))
            #      torch.Size([32, 3, 16, 64, 64]) <class 'torch.Tensor'>
            # print("step 2")

            pf = dis(fake)
            # print("step 3")
            dis_loss = torch.mean(-pr) + torch.mean(pf)
            disOpt.zero_grad()
            dis_loss.backward()
            disOpt.step()
            # update generator
            genOpt.zero_grad()
            fake = gen(torch.rand((batch_size, 2000), device='cuda')*2-1)
            # print("step 4")
            pf = dis(fake)
            gen_loss = torch.mean(-pf)
            gen_loss.backward()
            genOpt.step()
            gen_loss_metric.append(gen_loss)
            dis_loss_metric.append(dis_loss)

            if iteration % 5 == 0:
                for module in list(dis.model3d.children()) + [dis.conv2d]:
                    if type(module) == nn.Conv3d or type(module) == nn.Conv2d:
                        module.weight.data = singular_value_clip(module.weight)
                    elif type(module) == nn.BatchNorm3d:
                        gamma = module.weight.data
                        std = torch.sqrt(module.running_var)
                        gamma[gamma > std] = std[gamma > std]
                        gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                        module.weight.data = gamma

            # except:

        #     print('ens', frames, flush = True)
        #     err=+1
        # iteration += 1

        if(epoch%1 == 0):
            genSamples(gen, epochsx = 1 + epoch)
            torch.save(gen.state_dict(), 'gen' + str(1+epoch) + '.pt')
            torch.save(dis.state_dict(), 'disc' + str(1+epoch) + '.pt')

        # break

        #     pbar.update(len(data))
        # pbar.close()
        
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

gen, dis, errors, lg, ld = train_loop(epochs, gen, dis, videosDataset, gen_lr, dis_lr)


print('number of errors encountered :', errors)
print('gen-loss :', lg)
print('dis-loss :', ld)

torch.save(gen,'genf.pt')
torch.save(dis, 'disf.pt')


# ft = np.zeros([17, 3, 360, 360])
# ft = ft[:16]
# ft.shape


genSamples(gen)
