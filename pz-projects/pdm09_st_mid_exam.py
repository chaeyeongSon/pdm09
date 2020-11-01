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

st.header("수업시간에 배운 것 streamlit에 적용해보기!")
correlations = df.corr(method = 'pearson')
fig, ax = plt.subplots(1,1,figsize=(10,8))
img = ax.imshow(correlations, cmap='coolwarm',interpolation='nearest')  # 'hot'
ax.set_xticklabels(df.columns)
plt.xticks(rotation=90) # x축 글자들이 세로로 배치돼 겹침이 없어진다.
ax.set_yticklabels(df.columns)
fig.colorbar(img)
st.pyplot()
st.markdown("* * *")

plt.rcParams['figure.figsize'] = [12, 10]
df.hist()
st.pyplot()
st.markdown("* * *")

df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
st.pyplot()
st.markdown("* * *")

df.plot(kind= 'box', subplots=True, layout=(3,3), sharex=False, sharey=False)
st.pyplot()
st.markdown("* * *")

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='coolwarm' )
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
names = df.columns
ax.set_xticklabels(names,rotation=90) # Rotate x-tick labels by 90 degrees 
ax.set_yticklabels(names)
st.pyplot()
st.markdown("* * *")

from pandas.plotting import scatter_matrix

scatter_matrix(df)
st.pyplot()
st.markdown("* * *")

sns.heatmap(correlations, 
        xticklabels=df.columns,
        yticklabels=df.columns,
        vmin= -1, vmax=1.0)
st.pyplot()
st.markdown("* * *")

sns.pairplot(df)
st.pyplot()
st.markdown("* * *")

bar_labels = df.columns
plt.bar(bar_labels, df.mean(0), yerr=df.std(0), color='rgbcy')
st.pyplot()
st.markdown("* * *")

st.header("교수님께서 알려주신 것을 바탕으로 만들어보기!")
st.subheader('Outcome이 1(환자), 0(정상인)을 DB와 noDB로 바꾸기')
df['Outcome'] = df['Outcome'].apply(lambda x: 'DB' if x == 1 else 'noDB')
st.write(df['Outcome'])

st.subheader('당뇨병 환자와 당뇨병이 아닌 사람 수')
classes=df.Outcome
no, yes=classes.value_counts()
st.write('non-diabetes:',no)
st.write('diabetes:',yes)

st.subheader('당뇨병 환자와 당뇨병이 아닌 사람을 BAR-GRAPH로 그리기')
fig = plt.figure(figsize=(4,3))
sns.countplot(classes)
st.write(fig)
st.markdown("* * *")

st.subheader('당뇨병 환자와 당뇨병이 아닌 사람들을 각 항목과 연관이 있는지 boxplot으로 알아보기')
fig = plt.figure(figsize=(10, 10))

plt.subplot(331)
sns.boxplot(x = df['Outcome'], y = df['Pregnancies'])

plt.subplot(332)
sns.boxplot(x = df['Outcome'], y = df['Glucose'])

plt.subplot(333)
sns.boxplot(x = df['Outcome'], y = df['BloodPressure'])

plt.subplot(334)
sns.boxplot(x = df['Outcome'], y = df['SkinThickness'])

plt.subplot(335)
sns.boxplot(x = df['Outcome'], y = df['Insulin'])

plt.subplot(336)
sns.boxplot(x = df['Outcome'], y = df['BMI'])

plt.subplot(337)
sns.boxplot(x = df['Outcome'], y = df['DiabetesPedigreeFunction'])

plt.subplot(338)
sns.boxplot(x = df['Outcome'], y = df['Age'])

st.write(fig)