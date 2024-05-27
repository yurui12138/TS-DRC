import os
import sys

from utils import Metric

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from models.loss_dist import DRCLoss


def Trainer(model_ts, model_image, model_optimizer_ts, model_optimizer_image, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler_ts = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer_ts, 'min')
    scheduler_image = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer_image, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model_ts, model_image, model_optimizer_ts, model_optimizer_image, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, label_pre, labels_ture = model_evaluate(model_ts, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler_ts.step(valid_loss)
            scheduler_image.step(valid_loss)
        accuracy, macro_precision, macro_recall, weighted_f1 = Metric(labels_ture, label_pre)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\t | \tValid MF1     : {weighted_f1:2.4f}\t | \tValid precision     : {macro_precision:2.4f}\t | \tValid recall     : {macro_recall:2.4f}\n')
        if epoch % 10 == 0:
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'model_state_dict': model_ts.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last' + str(epoch) +'.pt'))

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model_ts.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, label_pre, labels_ture = model_evaluate(model_ts, test_dl, device, training_mode)
        accuracy, macro_precision, macro_recall, weighted_f1 = Metric(labels_ture, label_pre)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\t | \tTest MF1     : {weighted_f1:2.4f}\t | \tTest precision     : {macro_precision:2.4f}\t | \tTest recall     : {macro_recall:2.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model_ts, model_image, model_optimizer_ts, model_optimizer_image, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model_ts.train()
    model_image.train()

    logits_dict = {}

    for batch_idx, (data, labels, Markov, Gramian) in enumerate(tqdm(train_loader)):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        Markov, Gramian = Markov.float().to(device), Gramian.float().to(device)

        # optimizer
        model_optimizer_ts.zero_grad()
        model_optimizer_image.zero_grad()

        if training_mode == "self_supervised":
            prediction_ts, feature_ts = model_ts(data)
            # batch,dim     batch,dim       batch,dim
            feature_Markov = model_image(Markov)
            feature_Gramian = model_image(Gramian)
        else:
            outputs = model_ts(data)


        # compute loss
        if training_mode == "self_supervised":
            nt_xent_criterion = DRCLoss(device, config.batch_size_pretrain, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = nt_xent_criterion(feature_ts, feature_Markov, feature_Gramian)
        else: # supervised training or fine tuining
            predictions, feature_ts = outputs
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())


        total_loss.append(loss.item())
        loss.backward()
        model_optimizer_ts.step()
        model_optimizer_image.step()

    total_loss = torch.tensor(total_loss).mean()

    # print(pn_count)

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model_ts, test_dl, device, training_mode):
    model_ts.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, Markov, Gramian in test_dl:
            data, labels, Markov, Gramian = data.float().to(device), labels.long().to(device), Markov.float().to(device), Gramian.float().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                outputs = model_ts(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = outputs
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
