import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os

output_dir = '/home/yurui/pycharm/project/TS-DRC-main/data/Swat'

def preprocess(swat, row_mask_):
    swat_data = swat.iloc[:, :-2].copy()
    swat_label = swat.iloc[:, -1].copy()
    swat_rows = swat_data.shape[0]
    swat_cols = swat_data.shape[1]

    swat_data = swat_data.values.T
    swat_label = swat_label.values.T

    split_length = 128
    swat_data = swat_data[:, :split_length * (swat_rows // split_length)]
    swat_label = swat_label[:split_length * (swat_rows // split_length)]


    row_mask = []
    for i in range(len(swat_data)):
        lin_1 = swat_data[i][0]
        lin_2 = swat_data[i][:split_length]
        row_mask.append(np.all(lin_2 == lin_1))
    row_mask = np.array(row_mask)
    if row_mask_ != '':
        row_mask = row_mask_
    swat_data = swat_data[~row_mask]

    splits_samples = np.split(swat_data, swat_data.shape[1] // split_length, axis=1)
    splits_labels = np.split(swat_label, swat_label.shape[0] // split_length, axis=0)

    samples = np.stack(splits_samples, axis=0)
    labels = np.stack(splits_labels, axis=0)
    labels = np.where(np.any(labels != 0, axis=1), 1, 0)
    return samples, labels, row_mask


swat_train = pd.read_csv("SWaT_train.csv", skiprows=1)
swat_test = pd.read_csv("SWaT_test.csv", skiprows=1)

row_mask = ''
X_train, y_train, row_mask = preprocess(swat_train, row_mask)
X_test, y_test, _ = preprocess(swat_test, row_mask)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.15, random_state=42)


dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))





