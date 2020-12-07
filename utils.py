import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
import scikitplot.metrics as skplt
from PIL import Image
import os, json

#LOAD DATA
def load_datasets(datadir, normalise, batchsize=64, use_original_validation_set=True, use_test_set=False):

    #TRAINING SET
    train_dataset = datasets.ImageFolder(datadir + '/train',
                                         transforms.Compose([transforms.RandomResizedCrop(224),
                                                             #transforms.Resize([224,224]),
                                                             #transforms.RandomHorizontalFlip(0.5),
                                                             #transforms.RandomVerticalFlip(0.5),
                                                             #transforms.RandomRotation(45),
                                                             transforms.ToTensor(),
                                                             normalise]))
    # oversampling
    train_target = train_dataset.targets
    train_class_sample_count = np.unique(train_target, return_counts=True)[1]
    train_weight = 1. / train_class_sample_count
    train_samples_weight = torch.from_numpy(train_weight[train_target])
    train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, sampler=train_sampler)#shuffle=True)




    val_loader = None
    if(use_test_set):
        datadir = '/media/ext/Projects/ROP/data/images/split_4class/split_18_April_testset_new'
        # VALIDATION SET
        val_dataset = datasets.ImageFolder(datadir,
                                           transforms.Compose([#transforms.Resize([224, 224]),
                                                               transforms.Resize(256),
                                                               transforms.CenterCrop(224),
                                                               transforms.ToTensor(),
                                                               normalise]))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize)
    else:
        # VALIDATION SET
        val_dataset = datasets.ImageFolder(datadir + '/val',
                                           transforms.Compose([#transforms.Resize([224, 224]),
                                                               transforms.Resize(256),
                                                               transforms.CenterCrop(224),
                                                               transforms.ToTensor(),
                                                               normalise]))

        val_sampler = None
        if(not use_original_validation_set):
            val_target = val_dataset.targets
            val_class_sample_count = np.unique(val_target, return_counts=True)[1]
            val_weight = 1. / val_class_sample_count
            val_samples_weight = torch.from_numpy(val_weight[val_target])
            val_sampler = WeightedRandomSampler(val_samples_weight, len(val_samples_weight))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, sampler=val_sampler)

    return train_loader, val_loader

#HELPER FUNCTIONS
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_roc_curve(labels, output):
    skplt.plot_roc(labels, output, plot_micro=False)
    plt.show()

def plot_confusion_matrix(loader, labels,class_preds):
    skplt.plot_confusion_matrix(labels, class_preds)
    class_names = loader.dataset.classes
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.show()

    skplt.plot_confusion_matrix(labels, class_preds, normalize=True)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.show()