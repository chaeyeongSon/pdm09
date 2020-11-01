import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

st.header('기초') 
st.subheader('텍스트 출력') 
## Title
st.title('Streamlit Tutorial')
## Header/Subheader
st.header('This is header')
st.subheader('This is subheader')
## Text
st.text("Hello Streamlit! 이 글은 튜토리얼 입니다.")
st.markdown("* * *")

st.subheader('Markdown') 
## Markdown syntax
st.markdown("# This is a Markdown title")
st.markdown("## This is a Markdown header")
st.markdown("### This is a Markdown subheader")
st.markdown("- item 1\n"
            "   - item 1.1\n"
            "   - item 1.2\n"
            "- item 2\n"
            "- item 3")
st.markdown("1. item 1\n"
            "   1. item 1.1\n"
            "   2. item 1.2\n"
            "2. item 2\n"
            "3. item 3")
st.markdown("* * *")

st.subheader('Latex') 
## Latex
st.latex(r"Y = \alpha + \beta X_i")
## Latex-inline
st.markdown(r"회귀분석에서 잔차식은 다음과 같습니다 $e_i = y_i — \hat{y}_i$")
st.markdown("* * *")

st.subheader('메세지와 에러메세지, 예외처리 메세지') 
## Error/message text
st.success("Successful")
st.info("Information!")
st.warning("This is a warning")
st.error("This is an error!")
st.exception("NameError(‘Error name is not defined’)")
st.markdown("* * *")

st.subheader('데이터프레임과 테이블 출력.') 
## Load data
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris['target']
iris_df['target'] = iris_df['target'].apply(lambda x: 'setosa' if x == 0 else ('versicolor' if x == 1 else 'virginica'))

## Return table/dataframe
# table
st.table(iris_df.head())

# dataframe
st.dataframe(iris_df)
st.write(iris_df)

st.sidebar.header('EDA: Pima diabetes') 
st.sidebar.subheader('Visualization using seaborn and plotly')
st.markdown("* * *")
# ##Show image
# from PIL import Image
# img = Image.open("files/example_cat.jpeg")
# st.image(img, width=400, caption="Image example: Cat")
# ## Show videos
# vid_file = open("files/example_vid_cat.mp4", "rb").read()
# st.video(vid_file, start_time=2)
# ## Play audio file.
# audio_file = open("files/loop_w_bass.mp3", "rb").read()
# st.audio(audio_file, format="audio/mp3", start_time=10)

st.subheader('위젯-체크박스') 
## Checkbox
if st.checkbox("Show/Hide"):
 st.write("체크박스가 선택되었습니다.")

st.subheader('위젯-라디오버튼') 
## Radio button
status = st.radio("Select status.", ("Active", "Inactive"))
if status == "Active":
    st.success("활성화 되었습니다.")
else:
    st.warning("비활성화 되었습니다.")

st.subheader('위젯-드랍다운 선택') 
## Select Box
occupation = st.selectbox("직군을 선택하세요.",
 ["Backend Developer",
 "Frontend Developer",
 "ML Engineer",
 "Data Engineer",
 "Database Administrator",
 "Data Scientist",
 "Data Analyst",
 "Security Engineer"])
st.write("당신의 직군은 ", occupation, " 입니다.")

st.subheader('위젯-드랍다운 다중 선택') 
## MultiSelect
location = st.multiselect("선호하는 유투브 채널을 선택하세요.",
                          ("운동", "IT기기", "브이로그",
                           "먹방", "반려동물", "맛집 리뷰"))
st.write(len(location), "가지를 선택했습니다.")

st.subheader('위젯-슬라이더') 
## Slider
level = st.slider("레벨을 선택하세요.", 1, 5)

st.subheader('위젯-버튼') 
## Buttons
if st.button("About"):
 st.text("Streamlit을 이용한 튜토리얼입니다.")
st.markdown("* * *")

st.subheader('텍스트 입력') 
# Text Input
first_name = st.text_input("Enter Your First Name", "Type Here ...")
if st.button("Submit", key='first_name'):
    result = first_name.title()
    st.success(result)

# Text Area
message = st.text_area("메세지를 입력하세요.", "Type Here ...")
if st.button("Submit", key='message'):
    result = message.title()
    st.success(result)
st.markdown("* * *")

st.subheader('날짜와 시간 입력') 
 ## Date Input
import datetime
today = st.date_input("날짜를 선택하세요.", datetime.datetime.now())
the_time = st.time_input("시간을 입력하세요.", datetime.time())
st.markdown("* * *")

st.subheader('코드와 JSON 출력') 
## Display Raw Code — one line
st.subheader("Display one-line code")
st.code("import numpy as np")
# Display Raw Code — snippet
st.subheader("Display code snippet")
with st.echo():
 # 여기서부터 아래의 코드를 출력합니다.
 import pandas as pd
 df = pd.DataFrame()
## Display JSON
st.subheader("Display JSON")
st.json({"name" : "민수", "gender":"male", "Age": 29})

st.markdown("* * *")
st.markdown("* * *")

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


