import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os

output_dir = '/home/yurui/pycharm/project/TS-DRC-main/data/Wadi'

def preprocess(wadi, row_mask_):
    if row_mask_ == '':
        wadi_data = wadi.iloc[:, :-2].copy()
    else:
        wadi_data = wadi.iloc[:, :-1].copy()
    wadi_label = wadi.iloc[:, -2].copy()
    wadi_rows = wadi_data.shape[0]
    wadi_cols = wadi_data.shape[1]

    wadi_data = wadi_data.values.T
    wadi_label = wadi_label.values.T

    split_length = 128
    wadi_data = wadi_data[:, :split_length * (wadi_rows // split_length)]
    wadi_label = wadi_label[:split_length * (wadi_rows // split_length)]

    row_mask = []
    for i in range(len(wadi_data)):
        lin_1 = wadi_data[i][0]
        lin_2 = wadi_data[i][:split_length]
        row_mask.append(np.all(lin_2 == lin_1))
    row_mask = np.array(row_mask)
    if row_mask_ != '':
        row_mask = row_mask_
    wadi_data = wadi_data[~row_mask]

    splits_samples = np.split(wadi_data, wadi_data.shape[1] // split_length, axis=1)
    splits_labels = np.split(wadi_label, wadi_label.shape[0] // split_length, axis=0)

    samples = np.stack(splits_samples, axis=0)
    labels = np.stack(splits_labels, axis=0)
    labels = np.where(np.any(labels != 0, axis=1), 1, 0)
    return samples, labels, row_mask

wadi_train = pd.read_csv("WADI_train.csv", skiprows=1)
wadi_test = pd.read_csv("WADI_test.csv", skiprows=1)

row_mask = ''
X_test, y_test, row_mask = preprocess(wadi_test, row_mask)
X_train, _, row_mask = preprocess(wadi_train, row_mask)


X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.15, random_state=42)


dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(np.full(len(X_train), 1e-5))
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))





