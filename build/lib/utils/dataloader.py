
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np

class videosDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.annotations = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

        ll = sorted(list(self.annotations.iloc[:,1].unique()))
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

        vid, fc = self.FrameCapture(vid_path)

        if self.transform:
            vid = self.transform(vid)

        return (vid, vid_label, fc)

    def FrameCapture(self, path):

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