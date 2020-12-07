import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc
import scikitplot.metrics as skplt
from PIL import Image
import os, json
from lime import lime_image
from skimage.segmentation import mark_boundaries
from utils import *
from simple_cnn import *

#TRAIN AND EVALUATE FUNCTION
def train(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0
    model.train()

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)

    return epoch_loss / len(data_loader.dataset)

def evaluate(model, data_loader, criterion, device):
    epoch_loss = 0
    model.eval()
    labels_agg = []
    class_preds_agg = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)
            loss = criterion(output, labels)
            epoch_loss += loss.item() * inputs.size(0)

            output_exp = torch.exp(output)
            labels_agg += labels.cpu().detach().tolist()
            top_p, top_class = output_exp.topk(1, dim=1)
            class_preds_agg += list(np.ravel(top_class.cpu().detach().numpy()))
    cf = confusion_matrix(labels_agg, class_preds_agg)
    print(cf)
    print('Accuracy: ', (cf[0,0]+cf[1,1]+cf[2,2]+cf[3,3])/np.sum(cf))
    meanclassacc = []
    for i in range(4):
        foo = cf[i,i]/np.sum(cf[i])
        meanclassacc.append(foo)
        print('Class ', i, ' accuracy: ', foo)
    print('mean class acc: ', np.mean(meanclassacc))

    return epoch_loss / len(data_loader.dataset)


def main(is_training=True, continue_run = True, use_original_validation_set=False, use_test_set=False):
    #HYPERPARAMS
    datadir = '/media/ext/Projects/ROP/data/images/split_4class/split_18_April_trainval'
    batchsize = 64
    #normalise = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    learning_rate = 3e-4
    modelname = 'densenet_2'


    best_valid_loss = float('inf')
    best_epoch = 0
    train_losses = []
    valid_losses = []
    patience = 12  # how many epochs of increase in val loss to observe before stopping

    #LOAD DATA, CREATE MODEL ETC.
    print('loading data...')
    trainloader, validloader = load_datasets(datadir, normalise, batchsize, use_original_validation_set, use_test_set)
    print('finished loading data')
    #model = SimpleNet()
    model = models.densenet169(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1664, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512,128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Linear(128, 4),
                                     nn.LogSoftmax(dim=1))


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model.to(device)
    criterion.to(device)


    #TRAINING LOOP
    if(is_training):

        #continue training a saved model
        if(continue_run):
            model.load_state_dict(torch.load('/media/ext/Projects/ROP/Rahul/' + modelname + '.pt'))

        print('begin training...')
        for epoch in range(1000):
            start_time = time.time()

            train_loss = train(model, trainloader, optimizer, criterion, device)
            valid_loss = evaluate(model, validloader, criterion, device)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), '/media/ext/Projects/ROP/Rahul/' + modelname + '.pt')

            print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | {time.ctime()}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

            if (epoch - best_epoch) > patience:
                print(f'Early stopping at epoch {epoch}')
                break
    else: #load and evaluate model
        model.load_state_dict(torch.load('/media/ext/Projects/ROP/Rahul/' + modelname + '.pt'))
        model.eval()

        #for i, loader in enumerate([trainloader, validloader]):
        loader = validloader
        correct = 0
        total = 0
        labels_agg = []
        output_agg = []
        class_pred_agg = []
        print("Beginning model evaluation...")
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                output = torch.exp(output)

                #gather labels and predictions(both class and scores) for entire dataset
                #to create confusion matrix and ROC curve
                output_agg += list(output.cpu().detach().numpy())
                labels_agg += labels.cpu().detach().tolist()
                top_p, top_class = output.topk(1, dim=1)
                class_pred_agg += list(np.ravel(top_class.cpu().detach().numpy()))
                equals = top_class == labels.view(*top_class.shape)
                correct += torch.sum(equals.type(torch.FloatTensor)).item()
                total += len(equals)
        #print('training set') if i == 0 else print('validation set')
        print('Accuracy: ', correct / total)
        plot_confusion_matrix(loader, labels_agg, class_pred_agg)
        plot_roc_curve(labels_agg, output_agg)
        print('\n')



if __name__ == '__main__':
    #is_training = True means we are training a model, = False means we are evaluating a saved model
    #use_original_validation_set = True means we use the original validation set as is, without any over/undersampling.
    #  - We may want to set this var to False when training so that the val loss considers all classes equally, but
    #  - set it to true when evaluating a saved model, to produce a final confusion matrix or ROC curve
    main(is_training=False, continue_run = False, use_original_validation_set=True, use_test_set=True)

