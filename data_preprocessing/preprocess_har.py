'''
HAR数据集是6分类任务，因为志愿者进行六项活动（WALKING、WALKING_UPSTAIRS、WALKING_DOWNSTAIRS、SITTING、STANDING、LAYING）
所以记录的时序数据就是分别对应这6种活动的状态以及变化趋势
128是序列长度。因为每个样本数据是以2.56秒这一固定的时间窗口选取数据，也就是128个采样点的数据为一段。
9可以当做是传感器数量。因为9代表来自不同传感器的不同指标的数据，简单理解的话，可以直接当做9个传感器。类似是9维
'''

from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np

data_dir = '/home/yurui/pycharm/project/TS-TCC-main/data/HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset'
output_dir = './HAR'

subject_data = np.loadtxt(f'{data_dir}/train/subject_train.txt')
# Samples
train_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_x_train.txt')
train_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_y_train.txt')
train_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_z_train.txt')
train_gyro_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_x_train.txt')
train_gyro_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_y_train.txt')
train_gyro_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_z_train.txt')
train_tot_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_x_train.txt')
train_tot_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_y_train.txt')
train_tot_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_z_train.txt')

test_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_x_test.txt')
test_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_y_test.txt')
test_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_z_test.txt')
test_gyro_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_x_test.txt')
test_gyro_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_y_test.txt')
test_gyro_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_z_test.txt')
test_tot_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_x_test.txt')
test_tot_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_y_test.txt')
test_tot_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_z_test.txt')

# Stacking channels together data
train_data = np.stack((train_acc_x, train_acc_y, train_acc_z,
                       train_gyro_x, train_gyro_y, train_gyro_z,
                       train_tot_acc_x, train_tot_acc_y, train_tot_acc_z), axis=1)
X_test = np.stack((test_acc_x, test_acc_y, test_acc_z,
                      test_gyro_x, test_gyro_y, test_gyro_z,
                      test_tot_acc_x, test_tot_acc_y, test_tot_acc_z), axis=1)
# labels
train_labels = np.loadtxt(f'{data_dir}/train/y_train.txt')
train_labels -= np.min(train_labels)
y_test = np.loadtxt(f'{data_dir}/test/y_test.txt')
y_test -= np.min(y_test)


X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

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
