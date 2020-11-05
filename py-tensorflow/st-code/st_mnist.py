import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

from keras.datasets import mnist

# Load CIFAR10 dataset using tensorflow.keras
# Dividing data into training and test set
(trainX, trainy), (testX, testy) = mnist.load_data()
class_names = ["0","1","2","3","4","5","6","7","8","9"]

# Sidebar
st.sidebar.header('MNIST')
st.sidebar.subheader('Dataset of hand-written digits')
# Show a train random data
if st.sidebar.checkbox('Show a random train image from MNIST'):
    num = np.random.randint(0, trainX.shape[0])
    image = trainX[num]
    st.sidebar.image(image, caption=class_names[trainy[num]], width=192)
if st.sidebar.checkbox('Show a random test image from MNIST'):
    num = np.random.randint(0, testX.shape[0])
    image = testX[num]
    st.sidebar.image(image, caption=class_names[testy[num]], width=192)

# Main 
st.title('DL using CNN2D')
st.header('Dataset: cifar10')
#spending a few lines to describe our dataset
st.text("""Dataset of 60,000 28x28 gray training images, 
        labeled over 0 to 9, 
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
        mnist_train_dic = dict(zip(unique, counts))
        plt.bar(list(mnist_train_dic.keys()), mnist_train_dic.values(), color='g')
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
        mnist_test_dic = dict(zip(unique, counts))
        plt.bar(list(mnist_test_dic.keys()), mnist_test_dic.values(), color='g')
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