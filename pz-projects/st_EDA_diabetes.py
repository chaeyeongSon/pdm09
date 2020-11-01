import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.sidebar.header('EDA: Pima diabetes') 
st.sidebar.subheader('Visualization using seaborn and plotly')

# Get the data
df = pd.read_csv("https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/diabetes.csv")   # df에 github에 있는 csv파일을 읽어온다.
df['Outcome'] = df['Outcome'].apply(lambda x: 'DM' if x == 1 else 'noDM')   # Outcome에서 0은 noDM으로 1은 DM으로 바꿔준다.

classes=df.Outcome
noDB,DB=classes.value_counts()  # 당뇨병 환자인 사람과 아닌 사람을  count해준다.
st.sidebar.write('non-diabetes(noDM):',noDB)
st.sidebar.write('diabetes(DM):',DB)
# st.sidebar.write('***')
fig0 = plt.figure(figsize=(4,3))
sns.countplot(classes, label='count', palette=dict(noDM = 'g', DM = 'r'))   # Outcome을 바 그래프로 그려준다.(noDM은 GREEN, DM은 RED)
st.sidebar.write(fig0)  # 그림을 그려줌

st.subheader(f'Data Information: shape = {df.shape}')   # df의 shape를 출력
st.write(f'##### features = {list(df.columns)[:-1]}')   # df의 columns(열 제목)를 리스트 형태로 출력(마지막 열은 빼고)
st.write(f'##### classes = {pd.unique(classes)}')   # 서로 다른 값을 출력해줌 -> DM, noDM
st.write('***')
# Show the data as a table (you can also use st.write(df))
st.dataframe(df)    # dataframe을 전체 출력해줌
# Get statistics on the data
# st.write(df.describe())
# Show the data as a chart.
# chart = st.line_chart(df)
    
# histogram with plotly
st.header("Histogram")  # 제목으로 Histogram
## multi-column layput
row0_1, row0_space2, row0_2 = st.beta_columns(  # 최신 버전
    (1, .1, 1)) # 100%, 10%, 100%

with row0_1: 
    hist_x = st.selectbox("Select a feature", options=df.columns, index=df.columns.get_loc("Pregnancies"))  # 여러 개 중에 하나를 선택하는 셀렉트 바를 만들어줌(옵션은 열 제목들, 기본 값으로 Pregnancies)
        
with row0_2: 
    bar_mode = st.selectbox("Select barmode", ["relative", "group"], 0) # 기본이 0(relative)

hist_bins = st.slider(label="Histogram bins", min_value=5, max_value=50, value=25, step=1, key='h1')
# hist_cats = df['Outcome'].sort_values().unique()
hist_cats = df[hist_x].sort_values().unique()   # histogram 카테고리 
hist_fig1 = px.histogram(df, x=hist_x, nbins=hist_bins, 
                         title="Histogram of " + hist_x,
                         template="plotly_white", 
                         color='Outcome',   # 색을 Outcome으로 구분
                         barmode=bar_mode, 
                         color_discrete_map=dict(noDM = 'green', DM = 'red'),  
                         category_orders={hist_x: hist_cats}) 
st.write(hist_fig1)


# boxplots
st.header("Boxplot")
st.subheader("With a categorical variable - Outcome [noDM, DM]")
## multi-column layput
row1_1, row1_space2, row1_2 = st.beta_columns(
    (1, .1, 1))

with row1_1: 
    box_x = st.selectbox("Boxplot variable", options=df.columns, index=df.columns.get_loc("Age"))
        
with row1_2: 
    box_cat = st.selectbox("Categorical variable", ["Outcome"], 0)

st.write("Hint - try comparison w.r.t Catagories")
box_fig = px.box(df, x=box_cat, y=box_x, title="Box plot of " + box_x, color='Outcome', 
                        color_discrete_map=dict(noDM = 'green', DM = 'red'), 
                        template="plotly_white") #, category_orders={"pos_simple": ["PG", "SG", "SF", "PF", "C"]})
st.write(box_fig)

# Correlations
## multi-column layput
st.header("Correlations")
row2_1, row2_space2, row2_2 = st.beta_columns((1, .1, 1))

with row2_1: 
    corr_x = st.selectbox("Correlation - X variable", options=df.columns, index=df.columns.get_loc("Age"))
        
with row2_2: 
    corr_y = st.selectbox("Correlation - Y variable", options=df.columns, index=df.columns.get_loc("Pregnancies"))

corr2_fig = px.scatter(df, x=corr_x, y=corr_y, 
                            color='Outcome', 
                            color_discrete_map=dict(noDM = 'green', DM = 'red'), 
                            template="plotly_white")
st.write(corr2_fig)

# heatmap
st.subheader('Heatmap of correlation')
fig4 = plt.figure(figsize=(6,5))
sns.heatmap(df.corr(),annot=True, vmin=-1, vmax=1, cmap='coolwarm') # -1 : 반 상관성(차게), 1 : 상관성(따뜻하게)
st.pyplot(fig4)
# st.write(fig4)

# 출처: https://rfriend.tistory.com/409 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

# correlation heatmap
st.subheader('Heatmap of selected parameters')
fig5 = plt.figure(figsize=(5,4))
hmap_params = st.multiselect("Select parameters to include on heatmap", options=list(df.columns), default=[p for p in df.columns if "Outcome" not in p])    # 기본으로 Outcome을 뺀
sns.heatmap(df[hmap_params].corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
st.pyplot(fig5)
# st.write(hmap_fig)