import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
import time
from PIL import Image
import os
from lime import lime_image

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

if __name__ == '__main__':
    lime_explain()