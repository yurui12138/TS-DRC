# Time-Series Representation Learning via Dual Reference Contrasting
This is the code repository for the paper "Time-Series Representation Learning via Dual Reference Contrasting", which is accepted by the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24).
## Requirements:
- Python3.x
- Pytorch==1.7
- Numpy
- Sklearn
- Pandas
- openpyxl (for classification reports)
- mne=='0.20.7' (For Sleep-EDF preprocessing)
- mat4py (for Fault diagnosis preprocessing)
## Datasets
### Download datasets
We used four public datasets in this study, the datasets can be downloaded from the following URLs.:
- HAR: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
- ESP: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
- SWAT: https://itrust.sutd.edu.sg (this dataset needs to be applied for via email)
- WADI: https://itrust.sutd.edu.sg (this dataset needs to be applied for via email)

### Preparing datasets
The data should be in a separate folder called 'data' inside the project folder. Inside that folder, there should be separate subfolders for each dataset. Each subfolder should contain 'train.pt', 'val.pt', and 'test.pt' files. The structure of the data files should be in dictionary form as follows: train.pt = {'samples': data, 'labels': labels}, and similarly for val.pt and test.pt.

The details of preprocessing are as follows:
#### 1- HAR
After downloading the dataset and extracting the HAR dataset file, please move it to the data_preprocessing/HAR/ directory. 
Subsequently, run 'preprocess_har.py' script, which will generate the ‘train.pt’, ‘val.pt’, and ‘test.pt’ files. 
These files will be automatically saved in the data/HAR folder.

#### 2- ESP
After downloading the dataset and extracting the ESP dataset file, please move it to the 'data_preprocessing/ESP/' directory. 
Subsequently, run 'preprocess_esp.py' script, which will generate the ‘train.pt’, ‘val.pt’, and ‘test.pt’ files. 
These files will be automatically saved in the 'data/ESP' folder.

#### 3- SWAT
After downloading the dataset and extracting the SWAT dataset file, please move it to the 'data_preprocessing/SWAT/' directory. 
Subsequently, run 'preprocess_swat.py' script, which will generate the ‘train.pt’, ‘val.pt’, and ‘test.pt’ files. 
These files will be automatically saved in the 'data/SWAT' folder.

#### 4- WADI
After downloading the dataset and extracting the WADI dataset file, please move it to the 'data_preprocessing/WADI/' directory. 
Subsequently, run 'preprocess_wadi.py' script, which will generate the ‘train.pt’, ‘val.pt’, and ‘test.pt’ files. 
These files will be automatically saved in the 'data/WADI' folder.


### Configurations
The configuration files in the `config_files` folder should have the same name as the dataset folder name.
For example, for HAR dataset, the data folder name is `HAR` and the configuration file is `HAR_Configs.py`.
From these files, you can update the training parameters.

## Training TS-DRC 
You can select one of several training modes:
 - Random Initialization (random_init)
 - Supervised training (supervised)
 - Self-supervised training (self_supervised)
 - Fine-tuning the self-supervised model (fine_tune)
 - Training a linear classifier (train_linear)

The code also allows setting a name for the experiment and a name for separate runs in each experiment. Additionally, it allows the choice of a random seed value.

To use these options:
```
python main.py --experiment_description exp_HAR --run_description run_HAR --seed 123 --training_mode fine_tune --selected_dataset HAR
```
Note that the name of the dataset should be the same as the name inside the 'data' folder, and the training modes should match those specified above.

To train the model for the `fine_tune` modes, you have to run `self_supervised` first.

## Results
- The experiments are saved in the 'experiments_logs' directory by default (you can change this from the arguments as well).
- Each experiment will have a log file and a final classification report.

## Acknowledgement
If this work is useful in your research, please cite our paper.
```
@inproceedings{yu2024Time,
  title={Time-Series Representation Learning via Dual Reference Contrasting},
  author={Rui Yu, Yonghsun Gong, Shoujin Wang, Jiasheng Si, Xueping Peng, Bing Xu, and Wenpeng Lu},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  year={2024}
}
```

