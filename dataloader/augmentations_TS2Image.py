

import torch
from pyts.image import MarkovTransitionField, GramianAngularField, RecurrencePlot
from tsaug import TimeWarp, Crop, Quantize

mu = 0
sigma = 1

def AddNoise(series):
    noise = np.random.normal(mu, sigma, len(series))
    return series + noise

def simple_augument(sample):
    sample = sample.numpy()
    aug_shift = TimeWarp(n_speed_change=1)
    # aug_quantize = Quantize(n_levels=3)
    for i in range(len(data_noise)):
        sample[i] = AddNoise(sample[i])

    data_tw = [''] * len(sample)
    data_crop = [''] * len(sample)

    for i in range(len(sample)):
        data_tw[i] = aug_shift.augment(sample[i])


    for i in range(len(sample)):
        data_crop[i] = aug_quantize.augment(sample[i])

    return torch.Tensor(data_tw), torch.Tensor(data_crop)

def DataTransform(sample):
    batch = sample.shape[0]
    seq_len = sample.shape[-1]
    channel = sample.shape[1]
    samples = sample.view(-1,seq_len).numpy()

    sample_tw, sample_crop = simple_augument(sample)


    MTF = MarkovTransitionField()
    Markov_Images = MTF.transform(sample_tw)
    Markov_Images = torch.tensor(Markov_Images).unsqueeze(dim=1)


    GAF = GramianAngularField()
    Gramian_Images = GAF.transform(sample_crop)
    Gramian_Images = torch.tensor(Gramian_Images).unsqueeze(dim=1)



    return Markov_Images, Gramian_Images





