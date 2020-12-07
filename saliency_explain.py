import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
import time
from PIL import Image
from skimage.segmentation import mark_boundaries

#some of the saliency maps code was obtained from
#https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4

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
    saliency_explain()