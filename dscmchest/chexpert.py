import os
import copy
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread
from torch.utils.data import Dataset
import torch.nn.functional as F


def get_attr_max_min(attr):
    if attr == 'age':
        return 90, 18
    # elif attr == 'brain_volume':
    #     return 1629520, 841919
    # elif attr == 'ventricle_volume':
    #     return 157075, 7613.27001953125
    else:
        NotImplementedError


def norm(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] * 2 - 1  # [-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            batch[k] = batch[k].float().unsqueeze(-1)
    return batch


class CheXpertDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, columns=None, concat_pa=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices'
        ]

        self.samples = {
            'age': [],
            'sex': [],
            'finding': [],
            'x': [],
            'race': [],
        }
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = os.path.join(root, self.data.loc[idx, 'path_preproc'])

            disease = np.zeros(len(self.labels)-1, dtype=int)
            for i in range(1, len(self.labels)):
                disease[i-1] = np.array(self.data.loc[idx,
                                        self.labels[i]] == 1)

            finding = 0 if disease.sum() == 0 else 1

            self.samples['x'].append(img_path)
            self.samples['finding'].append(finding)
            self.samples['age'].append(self.data.loc[idx, 'age'])
            self.samples['race'].append(self.data.loc[idx, 'race_label'])
            self.samples['sex'].append(self.data.loc[idx, 'sex_label'])

        # self.samples = np.array(self.samples)
        # self.samples_copy = copy.deepcopy(self.samples)

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        # print(f'columns: {self.columns}')

        self.concat_pa = concat_pa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.samples.items()}

        # print(f'sample before: {sample}')
        sample['x'] = imread(sample['x']).astype(np.float32)[None, ...]

        for k, v in sample.items():
            sample[k] = torch.tensor(v)

        if self.transform:
            sample['x'] = self.transform(sample['x'])

        sample = norm(sample)
        # print(f'sample: {sample}')
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k in self.columns], dim=0)
        return sample
