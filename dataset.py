import torch
from pathlib import Path
import numpy as np
import glob
from torch.utils.data.dataset import Dataset
import pdb
import json
import os
import math
import cv2

class SpeedDataset(Dataset):
    def __init__(self, video_path, labels_path):
      self.labels = self.parse_labels(labels_path)
      self.frames = self.parse_frames(video_path)
      print(len(self.frames))
      print(len(self.labels))

    def parse_labels(self, labels_path):
      file = open(labels_path, 'r')
      lines = file.readlines()
      file.close()

      return lines

    def parse_frames(self, video_path):
      video = cv2.VideoCapture(video_path)

      frames = []
      while True:
        (grabbed, frame) = video.read()
        if not grabbed:
          break
        else:
          frames.append(frame)

      return frames

    def __getitem__(self, index):
      return [self.frames[index], self.labels[index]]

    def __len__(self):
      return len(self.labels)