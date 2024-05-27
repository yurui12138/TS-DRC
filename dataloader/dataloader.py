import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations_TS2Image import DataTransform

class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            self.Markov, self.Gramian = DataTransform(self.x_data[index])
            return self.x_data[index], self.y_data[index], self.Markov, self.Gramian
        else:
            return self.x_data[index], self.y_data[index], self.y_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_generator(data_path, configs, training_mode):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    if training_mode != 'self_supervised':
        batch_size = configs.batch_size_finetune

        train_dataset_sample = (valid_dataset['samples']).numpy()
        train_dataset_label = (valid_dataset['labels']).numpy()


        indexes_class = []
        for value in np.unique(train_dataset_label):
            indexes_class.extend(np.where(train_dataset_label == value)[0][:configs.fine_samples])
        indexes_class.sort()


        indexes_class = np.array(indexes_class)
        train_dataset_label = torch.tensor(train_dataset_label[indexes_class])
        train_dataset_sample = torch.tensor(train_dataset_sample[indexes_class])
        train_dataset['samples'] = train_dataset_sample
        train_dataset['labels'] = train_dataset_label
    else:
        batch_size = configs.batch_size_pretrain

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader