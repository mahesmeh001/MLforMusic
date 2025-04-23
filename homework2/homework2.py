#!/usr/bin/env python
# coding: utf-8

# # Homework 2  
# 
# The goal of this assignment is experiment with classification pipelines (in this case, for instrument classification) using spectrograms.

# In[41]:


# Set this yourself depending where you put the files
dataroot = "."
# On the autograder it should be here:
dataroot = "."


# In[42]:


# !pip install librosa
# !pip install torch
# !pip install glob
# !pip install numpy


# In[43]:


import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import random
import glob


# In[44]:


torch.use_deterministic_algorithms(True) # Try to make things less random, though not required


# In[45]:


audio_paths = glob.glob(dataroot + "/nsynth_subset/*.wav")
random.seed(0)
random.shuffle(audio_paths)


# In[46]:


if not len(audio_paths):
    print("You probably need to set the dataroot folder correctly")


# In[47]:


print(len(audio_paths))


# In[48]:


SAMPLE_RATE = 8000 # Very low sample rate, just so things run quickly
N_MFCC = 13
INSTRUMENT_MAP = {'guitar': 0, 'vocal': 1} # Only two classes (also so that things run quickly)
NUM_CLASSES = len(INSTRUMENT_MAP)

# If we used all the classes we would have:
# INSTRUMENT_MAP = {
#     'bass': 0, 'brass': 1, 'flute': 2, 'guitar': 3,
#     'keyboard': 4, 'mallet': 5, 'organ': 6, 'reed': 7,
#     'string': 8, 'synth_lead': 9, 'vocal': 10
# }


# 1. Extract prediction labels and construct waveforms
# 
# `extract_waveform()`
# 
# **Inputs**
# - `path`: A string that represents a path to the wav file
# 
# **Outputs**
# - `waveform`: an array containing the waveform; use librosa.load, remember to set the sample rate correctly
# 
# `extract_label()`
# 
# **Inputs**
# - `path'
# 
# **Outputs**
# - `label`: A integer that represents the label of the path (hint: look at the filename and make use of `INSTRUMENT_MAP`)

# In[49]:


def extract_waveform(path):
    # Your code here
    wave, sr = librosa.load(path, sr=SAMPLE_RATE)
    return wave 


# In[50]:


def extract_label(path):
    """Returns based on instrument map (guitar : 0), (vocal:1)"""
    # Your code here
    path = path.split('/')
    # print(path)
    return INSTRUMENT_MAP[path[2].split('_')[0]]


# In[51]:


waveforms = [extract_waveform(p) for p in audio_paths]
labels = [extract_label(p) for p in audio_paths]


# A few simple classifiers are provided. You don't need to modify these (though the autograder will *probably* work if you'd like to experiment with architectural changes)

# In[52]:


print(labels[:5])


# In[53]:


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * N_MFCC, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[54]:


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(nnF.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nnF.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nnF.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 2. Extract mfcc features
# 
# `extract_mfcc()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: a PyTorch float tensor that represents a concatenation of 13 mean values and 13 standard deviation values
# 
# **Process**
# - Extract feature using `librosa.feature.mfcc`; remember to set the sample rate and n_mfcc
# - Compute 13 mean and 13 standard deviation values
# - Concatenate them together

# In[55]:


def extract_mfcc(w):
    # Your code here:
    # load using librosa.feature.mfcc
    mfcc = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc = N_MFCC)
    # extract mean and 
    means = np.mean(mfcc, axis=1)
    
    stds = np.std(mfcc, axis=1)
    # concatenate
    features = np.concatenate([means, stds])
    
    return torch.FloatTensor(features)


# ## Note:
# 
# The autograder will test that your MFCC features are correct, and it will *also* use them within an ML pipeline. The test_suite can be used to run the full pipeline after you've implemented these functions. If you've implemented your features correctly this should "just work" and you'll be able to upload the trained; this is mostly here just so that you can see how the full pipeline works (which will be useful when you develop your own pipelines for Assignment 1)

# 3. Extract spectrograms
# 
# `extract_spec()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: a PyTorch float tensor that contains a spectrogram
# 
# **Process**
# - apply STFT to the given waveform
# - square the absolute values of the complex numbers from the STFT

# In[56]:


def extract_spec(w):
    # Your code here
    # load
    stft = librosa.stft(y=w)
    # take squared absolute values
    spec = np.abs(stft) ** 2
    
    
    return torch.FloatTensor(spec)


# 4. Extract mel-spectrograms
# 
# `extract_mel()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# - `n_mels`: number of mel bands
# - `hop_length`: hop length
# 
# **Outputs**
# - `feature`: A PyTorch Float Tensor that contains a mel-spectrogram
# 
# **Process**
# - generate melspectrograms with `librosa.feature.melspectrogram`; make sure to se the sample rate, n_mels, and hop_length
# - convert them to decibel units with `librosa.power_to_db`
# - normalize values to be in the range 0 to 1

# In[57]:


def extract_mel(w, n_mels = 128, hop_length = 512):
    # Your code here
    # load
    melspec = librosa.feature.melspectrogram(y=w, sr=SAMPLE_RATE, n_mels=n_mels, hop_length=hop_length)
    # convert to db
    mel_spec_db = librosa.power_to_db(melspec)
    
    # normalize
    mel_spec_db_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)

    
    return torch.FloatTensor(mel_spec_db_norm)


# 5. Extract constant-Q transform
# 
# `extract_q()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: A PyTorch Float Tensor that contains a constant-Q transform
# 
# **Process**
# - generate constant-Q transform with `librosa.cqt`; this one will need a higher sample rate (use 16000) to work

# In[58]:


def extract_q(w):
    # Your code here
    result = librosa.cqt(y=w, sr=16000)
    return torch.FloatTensor(result)

    # return torch.FloatTensor(result)


# 6. Pitch shift
# 
# `pitch_shift()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# - `n`: number of semitones to shift by (integer, can be positive or negative)
# 
# **Outputs**
# - `waveform`: a pitch-shifted waveform
# 
# **Process**
# - use `librosa.effects.pitch_shift`

# In[59]:


def pitch_shift(w, n):
    # Your code here
    return librosa.effects.pitch_shift(y=w, sr = SAMPLE_RATE, n_steps=n)


# In[60]:


# Code below augments the datasets

augmented_waveforms = []
augmented_labels = []

for w,y in zip(waveforms,labels):
    augmented_waveforms.append(w)
    augmented_waveforms.append(pitch_shift(w,1))
    augmented_waveforms.append(pitch_shift(w,-1))
    augmented_labels += [y,y,y]


# 7. Extend the model to work for four classes.
# 
# By making data augmentations, or modifying the model architecture, build a model with test accuracy > 0.93

# In[61]:


INSTRUMENT_MAP_7 = {'guitar_acoustic': 0, 'guitar_electronic': 1, 'vocal_acoustic': 2, 'vocal_synthetic': 3}


# In[62]:


NUM_CLASSES_7 = 4


# In[63]:


def extract_label_7(path):
    # Your code here
    pass


# In[64]:


# Select which feature function to use.
# Can be one of the existing ones (e.g. extract_mfcc), or you can write a new one.
feature_func_7 = None


# In[65]:


labels_7 = [extract_label_7(p) for p in audio_paths]


# In[66]:


# Select which model to use.
# Can use an existing model (e.g. MLPClassifier) or modify it.
# Note that you'll need to copy and (slightly) modify the existing class to handle 4 labels.
# model_7 = MLPClassifier_4classes()


# In[ ]:


# get_ipython().system('jupyter nbconvert homework2.ipynb --to python')

