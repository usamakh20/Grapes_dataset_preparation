import os
import glob
import torch
import shutil
import numpy as np
from collections import namedtuple
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

data_dir = 'data'
batched_data_dir = 'output_data'
varietals = ['CDY', 'CFR', 'CSV', 'SVB', 'SYH']
data_item = namedtuple('item', ['image', 'annotations'])

if os.path.exists(batched_data_dir) and os.path.isdir(batched_data_dir):
    shutil.rmtree(batched_data_dir)

os.makedirs(batched_data_dir)


class GrapesDataset(Dataset):

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
        """
        self.annotation_files = glob.glob(os.path.join(root_dir, '*.txt'))

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.annotation_files[idx].split('.')[0] + '.jpg'
        image = io.imread(img_path)
        annotations = np.loadtxt(self.annotation_files[idx])[:, 1:]
        sample = data_item(image, annotations)

        return sample


if __name__ == '__main__':

    dataset = GrapesDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=5,
                            shuffle=True, num_workers=0,
                            collate_fn=lambda batch: tuple(zip(*batch)))

    for i_batch, sample_batched in enumerate(dataloader):
        path = os.path.join(batched_data_dir, 'batch_' + str(i_batch), '')
        os.makedirs(path)
        for i in range(len(sample_batched[0])):
            io.imsave(path + 'img_' + str(i) + '.jpg', sample_batched[0][i])
            np.savetxt(path + 'annotation_' + str(i) + '.txt', sample_batched[1][i])
