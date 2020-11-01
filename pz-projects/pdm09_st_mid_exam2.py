import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get the data
df = pd.read_csv("https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/diabetes.csv")

st.subheader('Data Information:')
# Show the data as a table (you can also use st.write(df))
st.dataframe(df)
# Get statistics on the data
st.write(df.describe())

# Show the data as a chart.
chart = st.line_chart(df)

## mid-term practice
## EDA of diabetes.csv
# Your code here !!

# Draw histograms for all attributes 
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("* * *")


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y= df.iloc[:,-1].values
# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17,3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure',0, 122,72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    
    
    user_data = {'pregnancies': pregnancies,
              'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
              'BMI': BMI,
              'DPF': DPF,
                 'age': age
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()
st.subheader('User Input :')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )
prediction = RandomForestClassifier.predict(user_input)
st.subheader('Classification: ')
st.write(prediction)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']], df['Outcome'], test_size=0.3, random_state=109)
#Creating the model
logisticRegr = LogisticRegression(C=1)
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
#Saving the Model
pickle_out = open("logisticRegr.pkl", "wb") 
pickle.dump(logisticRegr, pickle_out) 
pickle_out.close()
