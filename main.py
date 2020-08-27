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
    # fpr, tpr, _ = roc_curve(labels, output)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

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

#simple CNN
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernelsize=5):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernelsize, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=16, kernelsize=3)
        self.unit2 = Unit(in_channels=16, out_channels=16, kernelsize=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,2))

        self.unit3 = Unit(in_channels=16, out_channels=32, kernelsize=3)
        self.unit4 = Unit(in_channels=32, out_channels=32, kernelsize=3)

        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,2))

        self.unit5 = Unit(in_channels=32, out_channels=64, kernelsize=3)
        self.unit6 = Unit(in_channels=64, out_channels=64, kernelsize=3)

        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2, 2))


        #Conv
        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit3, self.unit4, self.pool2,
                                 self.unit5, self.unit6, self.pool3)

        #batchnorm
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.BatchNorm1d(128)

        #FC
        self.fc = nn.Sequential(nn.Linear(28*28*64, 1024),
                                #self.bn1,
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(1024, 128),
                                #self.bn2,
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(128,4),
                                nn.LogSoftmax(dim=1))

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 28*28*64)
        output = self.fc(output)
        return output


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

            output_exp = torch.exp(output)#rrr
            labels_agg  += labels.cpu().detach().tolist()
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
    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Sequential(nn.Linear(512, 256),
    #                          #nn.BatchNorm1d(256),
    #                          nn.ReLU(),
    #                          nn.Dropout(0.3),
    #                          nn.Linear(256,128),
    #                          #nn.BatchNorm1d(128),
    #                          nn.ReLU(),
    #                          nn.Dropout(0.3),
    #                          nn.Linear(128,4),
    #                          nn.LogSoftmax(dim=1))
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Sequential(nn.Linear(2048, 512),
    #                          nn.BatchNorm1d(512),
    #                          nn.ReLU(),
    #                          #nn.Dropout(0.3),
    #                          nn.Linear(512,128),
    #                          nn.BatchNorm1d(128),
    #                          nn.ReLU(),
    #                          #nn.Dropout(0.3),
    #                          nn.Linear(128,4),
    #                          nn.LogSoftmax(dim=1))

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

            if True:#rrrvalid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), '/media/ext/Projects/ROP/Rahul/' + modelname + '.pt')
                #np.savetxt('/media/ext/Projects/ROP/Rahul/train_losses' + '_' + modelname + '.txt', train_losses)
                #np.savetxt('/media/ext/Projects/ROP/Rahul/valid_losses' + '_' + modelname + '.txt', valid_losses)

            print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | {time.ctime()}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

            if False:#rrr(epoch - best_epoch) > patience:
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




#LIME Explanation methods
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

class LimeModel:
    def __init__(self, model, preprocess_transform):
        self.model = model
        self.pt = preprocess_transform

    def batch_predict(self, images):
        model = self.model
        model.eval()
        batch = torch.stack(tuple(self.pt(i) for i in images), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)
        logsoft = model(batch)
        probs = torch.exp(logsoft)
        return probs.detach().cpu().numpy()


#code for the lime explanation was obtained from
#https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
def lime_explain():
    root = '/media/ext/Projects/ROP/data/images/split_4class/split_18_April_testset_new/'
    img = get_image(root + 'ungradeable/fa8fa02c-f64c-401e-9db6-2cf1275caca6.23.png')
    plt.imshow(img)
    plt.show()

    model = models.densenet169(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1664, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Linear(128, 4),
                                     nn.LogSoftmax(dim=1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    modelname = 'densenet_2'
    model.load_state_dict(torch.load('/media/ext/Projects/ROP/Rahul/' + modelname + '.pt'))
    model.eval()

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    limemodel = LimeModel(model, preprocess_transform)

    test_pred = limemodel.batch_predict([pill_transf(img)])
    print("predicition probs: ", test_pred.squeeze())

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             limemodel.batch_predict,
                                             top_labels = 1)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
    img_boundary = mark_boundaries(temp /255.0, mask)
    plt.imshow(img_boundary)
    plt.show()


#backprop-based saliency methods
# Preprocess the image
def preprocess(image):
    transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def deprocess(image):
    transform = T.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))
    plt.show()


#some of the saliency maps code was obtained from
#https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
def saliency_explain():
    root = '/media/ext/Projects/ROP/data/images/split_4class/split_18_April_testset_new/'
    img = get_image(root + 'ungradeable/fa8fa02c-f64c-401e-9db6-2cf1275caca6.23.png')
    pil_transform = get_pil_transform()
    img = pil_transform(img)
    plt.imshow(img)
    plt.show()

    model = models.densenet169(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1664, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Linear(128, 4),
                                     nn.LogSoftmax(dim=1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    modelname = 'densenet_2'
    model.load_state_dict(torch.load('/media/ext/Projects/ROP/Rahul/' + modelname + '.pt'))
    model.eval()

    X = preprocess(img)
    X = X.to(device)
    X.requires_grad_()
    scores = model(X)
    scores = torch.exp(scores)
    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]
    score_max.backward()
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    saliency = saliency.cpu()
    plt.imshow(saliency[0], cmap=plt.cm.hot)
    plt.show()



if __name__ == '__main__':
    #is_training = True means we are training a model, = False means we are evaluating a saved model
    #use_original_validation_set = True means we use the original validation set as is, without any over/undersampling.
    #   We may want to set this to False when training so that the val loss considers all classes equally, but
    #   set it to true when evaluating a saved model, to produce a final confusion matrix or ROC curve
    #main(is_training=False, continue_run = False, use_original_validation_set=True, use_test_set=True)

    #generate lime explanation
    #lime_explain()

    #generate saliency maps
    saliency_explain()