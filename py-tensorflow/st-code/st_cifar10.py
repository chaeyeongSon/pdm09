import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

from keras.datasets import cifar10

# Load CIFAR10 dataset using tensorflow.keras
# Dividing data into training and test set
(trainX, trainy), (testX, testy) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Sidebar
st.sidebar.header('CIFAR10')
st.sidebar.subheader('Dataset')
# Show a train random data
if st.sidebar.checkbox('Show a random train image from CIFAR10'):
    num = np.random.randint(0, trainX.shape[0])
    image = trainX[num]
    st.sidebar.image(image, caption=class_names[trainy[num].item()], width=192)

# Show a test random data
if st.sidebar.checkbox('Show a random test image from CIFAR10'):
    num = np.random.randint(0, testX.shape[0])
    image = testX[num]
    st.sidebar.image(image, caption=class_names[testy[num].item()], width=192)

# Main 
st.title('PDM09, 손채영')
st.header('Dataset: cifar10')
#spending a few lines to describe our dataset
st.text("""Dataset of 50000 32x32 color training images, 
        consist of 'vehicle' and 'animal'
        and 10,000 test images.""")

# Information of mnist dataset
if st.checkbox('Show images sizes'):
    st.write(f'##### X Train Shape: {trainX.shape}') 
    st.write(f'##### X Test Shape: {testX.shape}')
    st.write(f'##### Y Train Shape: {trainy.shape}')
    st.write(f'##### Y Test Shape: {testy.shape}')
st.write('***')

radio = st.radio("Select: ", ("Train", "Test"))
if radio == "Train":
    if st.checkbox('Show train set information'):
        unique, counts = np.unique(trainy, return_counts=True)
        cifar10_train_dic = dict(zip(unique, counts))
        plt.bar(list(cifar10_train_dic.keys()), cifar10_train_dic.values(), color='g')
        st.pyplot()
    if st.checkbox('Show 10 different image from the train set'):
        num_10 = np.unique(trainy, return_index=True)[1]
#       st.write(num_10)
        images = trainX[num_10]
        for i in range(len(images)):
            # define subplot
            plt.subplot(2,5,1 + i) #, sharey=False)
            # plot raw pixel data
            plt.imshow(images[i])
            plt.title(class_names[i])
            plt.xticks([])
            plt.yticks([])
        plt.suptitle("10 different train images", fontsize=18)
        st.pyplot()  # Warning
if radio == "Test":
    if st.checkbox('Show test set information'):
        unique, counts = np.unique(testy, return_counts=True)
        cifar10_test_dic = dict(zip(unique, counts))
        plt.bar(list(cifar10_test_dic.keys()), cifar10_test_dic.values(), color='g')
        st.pyplot()
    if st.checkbox('Show 10 different image from the test set'):
        num_10 = np.unique(testy, return_index=True)[1]
#       st.write(num_10)
        images = testX[num_10]
        for i in range(len(images)):
            # define subplot
            plt.subplot(2,5,1 + i) #, sharey=False)
            # plot raw pixel data
            plt.imshow(images[i])
            plt.title(class_names[i])
            plt.xticks([])
            plt.yticks([])
        plt.suptitle("10 different test images", fontsize=18)
        st.pyplot()  # Warning