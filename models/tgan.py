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

def genSamples(g, n=8):
    '''
    Generate an n by n grid of videos, given a generator g
    '''
    with torch.no_grad():
        s = g(torch.rand((n**2, 100), device='cuda')*2-1).cpu().detach().numpy()
    
    out = np.zeros((3, FRAMES_PER_VIDEO, 64*n, 64*n))
    # print(out.shape)
    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n+k, :, :, :, :]

    
    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    out = out.astype(int)
    clip = ImageSequenceClip(list(out), fps=20)
    clip.write_gif('sample.gif', fps=20)

class TemporalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a sequential model to turn one vector into 16
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
        x = x.view(-1, 100, 1)
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
        z_fast = z_fast.view(-1, 100)

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

gen.load_state_dict(torch.load('state_normal81000.ckpt')['model_state_dict'][0])





# genSamples(gen)
# def display_gif(fn):
#     from IPython import display
#     return display.HTML('<img src="{}">'.format(fn))
# display_gif("sample.gif")




import pandas as pd
df = pd.read_csv('result400.csv')
ll = df.iloc[:,1]





ll.tolist()
ll = set(ll)
dic = {}
for id, i in enumerate(ll):
    dic[i] = id
# dic





# import pandas as pd
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

# # ([17, 360, 360, 3])





# pwd





batch_size = 32
epochs = 5
gen_lr = 1e-4
dis_lr = 1e-4
device = 'cuda'

# frames per video = 16
# disOpt = 
# genOpt = 

csv_file = 'result400.csv'
root_dir = './'
rr = './'
data = videosDataset(csv_file, rr, transform = None)

train_loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)





l = [10, 20, 30]





sum(l)/len(l)





len(train_loader)





# from tqdm.notebook import tqdm





def train_loop(epochs, gen, dis, videosDataset, gen_lr, dis_lr):
    # criterion = nn.CrossEntropyLoss()
    genOpt = optim.Adam(gen.parameters(), lr = gen_lr)
    disOpt = optim.Adam(dis.parameters(), lr = dis_lr)
    gen_loss_metric = []
    dis_loss_metric = []
    err = 0
    for epoch in range(epochs):
        losses = []
        # pbar = tqdm(range(len(train_loader), desc="Batch:")
        for data, targets, frames in (train_loader):
            

            # print(type(real))
            # out_g = gen()
            # update discriminator
            # "real" sequence of imges comes from the dataset
            #  torch.Size([32, 3, 16, 64, 64]) <class 'torch.Tensor'>
            # pr = torch.zeros([32, 3, 16, 64, 64])
            try:

                
                data = data.to(device)
                targets = targets.to(device)
                pr = dis(data)
                # print("step 1")

                fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
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
                fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
                # print("step 4")
                pf = dis(fake)
                gen_loss = torch.mean(-pf)
                gen_loss.backward()
                genOpt.step()
                gen_loss_metric.append(gen_loss)
                dis_loss_metric.append(dis_loss)
            except:
                
                print('frames found in vid :', frames, flush = True)
                err=+1
                
        #     pbar.update(len(data))
        # pbar.close()
        
        print(f"Genrator Loss: {sum(gen_loss_metric)/len(gen_loss_metric)}", flush = True)
        print(f"Discriminator Loss: {sum(dis_loss_metric)/len(dis_loss_metric)}", flush = True)
        
    return (gen, dis, err, gen_loss_metric, dis_loss_metric)




gen2, dis2, errors, lg, ld = train_loop(epochs, gen, dis, videosDataset, gen_lr, dis_lr)


print('number of errors encountered :', errors)
print('gen-loss :', lg)
print('dis-loss :', ld)

torch.save(gen2,'gen.pt')
torch.save(dis2, 'dis.pt')




# ft = np.zeros([17, 3, 360, 360])
# ft = ft[:16]
# ft.shape


genSamples(gen2)


