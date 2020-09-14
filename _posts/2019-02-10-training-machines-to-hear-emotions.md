---
title: 'Training Machines to Hear our Emotions'
date: 2019-02-10
permalink: /posts/2019/02/training-machines-to-hear-our-emotions/
tags:
  -
---

[![Screen-Shot-2019-02-13-at-9-48-46-AM.png](https://i.postimg.cc/3xBYHckC/Screen-Shot-2019-02-13-at-9-48-46-AM.png)](https://postimg.cc/BjXdTNcX)

Human and machine interaction is becoming increasingly foundational to our society. From highly efficient automated call centers to surprisingly intimate companion machines, learning how to conduct life along side machines will be one of the major advancements and challenges of the 21st century. One of the principal components of fostering this cultural transition will be building machines that can predict our emotions, however irrational and incomprehensible they may sometimes seem. We generally understand machines, so it would be good if they generally understood us.

In this project, I built a classification model to predict positive or negative emotion in human speech with 83% accuracy/precision. In the grand scheme of things, 83% is not very impressive, but emotion is deeply complex and much of it remains shroud in mystery to this day. While I will continue to tune this model and hone its precision, I wanted to write about it to clarify my thoughts and share my findings.

However before I dive in to the details of the model, I want to preface that I carry two polarizing perceptions of this powerful technology of which I have only scratched the surface. The romantic within me feels that the subjectivity, mystery, and power of emotion is one of the last frontiers of the human experience left unspoiled. If we strip emotion down to synapses and waveform statistics, will that degrade the potency of art, love, and beauty? On the other end, the technologist in me envisions a future with human-centered AI, where our daily lives are full of interactions with emotionally perceptive machines that enhance the human experience.

### Tools
To complete this project, I used python, jupyter notebooks, postgreSQL, pandas, numpy, matplotlib, scikit-learn, Ipython, Librosa, pyAudioAnalysis, and XGBoost.

### Data sources
The data source for this project will be the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D). It can be found here:

This data contains ~7,500 .mp3 files containing recordings of actors saying the same 12 sentences with six different emotions. Other tables containing demographic data per actor and emotional tone per file were merged with audio files and cleaned in a SQL database.

### Feature extraction
Numerous spectral features were extracted from waveforms as arrays from the raw mp3 files using the Librosa library. To prepare features for modeling, the mean and standard deviation was computed for each feature. The final spectral features I included in my model were: chroma (standard deviation), contrast (mean), energy (mean), energy (standard deviation), MFCC (standard deviation), flatness (standard deviation), and zero cross rate (mean & standard deviation). As audio feature extraction can be used for a wide variety of problems, check out my code below and feel free to use it for any audio project.

```
import numpy as np
import pandas as pd
# Audio feature extraction
import librosa as lb
# Importing files
from os import listdir
from os.path import isfile, join
 
 
def get_files(path):
    '''Gets all files (as strings) from a directory into a list.'''
 
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    return filenames
 
get_files(your_path/CREMA-D/AudioMP3')
 
 
def extract_features(filename):
    '''Loading in audio files and extracting features. Features include:
        - mfcc = Mel-frequency cepstral coefficients. Used for vocals.
        - centroid = Spectral Centroid. Mean value of frequency form.
        - flatness = Spectral flatness. Noisy vs harmonic sound.
        - tempo = Spectral onset envelope. Describes rythm.
        - cens = Chroma Energy Normalized Statistics. Smooths frequency 
         windows for matching.
        - energy = Root Mean Square Energy. Computes energy of each frame.
        - melspec = Mel-Scaled Spectrogram.
        - contrast = Spectral Contrast.
        - tonnetz = Tonnetz. Computes tonal centroid features.
        - chroma = Chromagram from waveform.
    '''
 
    # load in file. y is the waveform, sr is the sampling rate.
    y, sr = lb.load(filename)
 
    # Short-time Fourier transformation.
    stft = np.abs(lb.stft(y))
    S, phase = lb.magphase(np.abs(stft))
 
    # all features
    mfcc = np.mean(lb.feature.mfcc(y=y, sr=sr))
    centroid = np.mean(lb.feature.spectral_centroid(y=y, sr=sr))
    flatness = np.mean(lb.feature.spectral_flatness(y=y, S=S))
    tempo = np.mean(lb.feature.tempogram(y=y, sr=sr))
    cens = np.mean(lb.feature.chroma_cens(y=y, sr=sr))
    energy = np.mean(lb.feature.rmse(y=y))
    melspec = np.mean(lb.feature.melspectrogram(y=y, sr=sr))
    contrast = np.mean(lb.feature.spectral_contrast(y=y, sr=sr))
    tonnetz = np.mean(lb.feature.tonnetz(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr))
 
    # Create a dict of features
    audio_features = {'filename': filename,
                     'mfcc': mfcc,
                     'centroid': centroid,
                     'flatness': flatness,
                     'tempo': tempo,
                     'cens': cens,
                     'energy': energy,
                     'melspec': melspec,
                     'contrast': contrast,
                     'tonnetz': tonnetz,
                     'chroma': chroma
                     }
    return audio_features
 
def make_df(extract_features):
    '''This function calls the extract_feature function on every file, 
     putting each audio file dict into a list and returns a df with each 
     dictionary as a row.'''
 
    data = []
    for i in filenames:
        data.append(extract_features(i))
    df_features = pd.DataFrame(data)
    df_features.to_csv('features_data.csv', sep=',', encoding='utf-8')
    return df_features
 
make_df(extract_features)

```

As this was my first pass working with audio files, I decided to simplify the model by dropping sadness and fear, which were very difficult to distinguish for both humans and machine. Also, I decided to streamline my classification by using a binary positive-negative classification. For positive emotion I combined neutral and happiness and negative emotion I combined anger and disgust.

### Model and Results

I wanted my model to take in variety of audio features and predict whether the recording could be classified as positive or negative. To do so, I implemented a variety of classification algorithms for this project, including Logistic Regression, K Nearest Neighbors, Random Forest, SVM, and XGBoost, all optimized with GridSearchCV. Parameters were tuned to optimize precision, which was my metric of concern so that my model limited false positives. XGBoost was chosen as my final model because it was the strongest performer across the board, with precision and accuracy at .83.

I chose to optimize my model for precision, which is a measure of what proportions of positive predictions were actually correct. This is important when predicting emotions because no one wants to be treated as if they were fine and happy when they were actually upset. Imagine the following scenario, imagine how it might go, and you’ll get the picture why we want to avoid false positives.

Automated call center bot: ‘Hello, how are you doing today?’
Person: ‘Thanks for asking, but I’ve been better.‘
Automated call center bot: ‘I’m glad to hear you’re doing well, how can I help you?’
Person: ‘I wanted to ask a question about your product, but it sounds like you’re misunderstanding me.’
Automated call center bot: ‘‘I’m thrilled that you want to talk about our product, please ask your question.’
Person (now disgruntled): ‘This is weird, are you even listening to me? Wait, are you a robot? Ugh, this company is so cheap they don’t even employ real people. I’m leaving a bad review.’

It’s unlikely this scenario would happen as automated respondents will likely have a natural language processing component that would catch text sentiment, but you get the point.

To better visualize the model performance, see this confusion matrix for predictions on my test data. Note the low occurrence of False Positives, which is what I wanted to avoid.

[![Screen-Shot-2019-02-13-at-10-27-04-AM.png](https://i.postimg.cc/7ZLQfsBN/Screen-Shot-2019-02-13-at-10-27-04-AM.png)](https://postimg.cc/F72x6ZXf)

Below is the ROC curve, along with the rest of the performance metrics on my model. The curve represents the models sensitivity to specificity ratio and the closer it is to the left and top border, the better its predictive power. The curve is very distinct from the 45 degree line in the center which represents random chance, so the model is in decent shape.

[![Screen-Shot-2019-02-13-at-11-21-16-AM.png](https://i.postimg.cc/3NNHjXYM/Screen-Shot-2019-02-13-at-11-21-16-AM.png)](https://postimg.cc/1nkTsN8M)

### Future Work

I originally set out to build a model that predicted two classes, positive or negative emotion. From my experience in emotional analysis, a ratio of positive:negative emotion is an incredibly powerful and efficient way to analyze affect. However, this may have not been the best approach. Notice in the image below how polar emotions like happiness and anger look fairly similar compared to neutral and sadness in terms of spectral flatness. This illustrates a positive class emotion (happiness) appearing more similar to a negative class emotion (anger) than its fellow positive emotions (neutral). Forcing spectral features from emotions that we intuitively group together in similar classes may not be optimal for machine learning, and predicting a class for each emotion on its own may increase overall performance.

[![Screen-Shot-2019-02-13-at-9-47-39-AM.png](https://i.postimg.cc/L8ywDKq7/Screen-Shot-2019-02-13-at-9-47-39-AM.png)](https://postimg.cc/18qvRdrc)

Too late in the scope of the project timeline, I discovered the high-level benefits of pyAudioAnalysis. Building my model from the ground up in Librosa was a great exercise, but pyAudioAnalysis provides high-level functions to extract features and construct high performance classification models with relative ease. With pyAudioAnalysis, I was able to train a multiclass SVM model on the same training data used before with just a few lines of code. The new model was able to predict the emotional class of my own recorded speech utterances with probabilities in the 0.70-0.80 range. This is a very promising future direction for my work in audio analysis.

Final Thoughts
While I will likely remain torn about emotionally perceptive AI for some time, its progress is inevitable. Instead being that guy who complained that the invention of writing would atrophy our memories or that shovels would weaken our hands, I can think of no better approach than trying to understand this new technology. I care deeply about this issue, and I am eager to take part in building it, to watch it develop from an informed perspective, and to play a role in discussing its ethical implications.

This is an amazing era to study the psychology and machine learning and I am so grateful to be somewhere in the middle. This intersection will greatly influence the way we interact with machines in the coming years, and I look forward to playing my role in shaping a meaningful future.
