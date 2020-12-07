# ROP detection

![Fundus image](https://miro.medium.com/max/1400/1*cFM4tWOcLeGjwxuqSNLytA.png)

This repository contains work done as part of my master's thesis at UCL on the Interpretability of Convolutional Neural Networks trained to detect Retinopathy of Prematurity, in collaboration with Moorfields Eye Hospital, London, UK.

1. `train.py` - Train a classifier
2. `utils.py` - Helper functions
3. `simple_cnn.py` - Baseline CNN model
4. `lime_explain.py` - Generate LIME explanations for classifier's predictions
5. `saliency_explain.py` - Generate backpropagation-based saliency maps for classifier's predictions
