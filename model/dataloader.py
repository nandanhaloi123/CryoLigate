from torch.utils.data import Dataset, DataLoader
import mrcfile
import numpy as np
import json
import torch

# Data loader
class EMdata(Dataset):
    def __init__(self, dataset_directory="../dataset/"):
        # set root directory for your dataset
        self.dataset_directory = dataset_directory

        # read json file with annotations
        annotations_file = open(self.dataset_directory + "all_data.json")
        self.annotations = json.load(annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        image_filename = self.annotations[str(i)]['file_name']
        image_path = self.dataset_directory + image_filename
        # print(image_path)
        # image = img.imread(image_path)
        image = torch.tensor(mrcfile.read(image_path)).unsqueeze(0) # adding one channel with zero values

        return image