import streamlit as st
from config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import pickle

Datapath = 'datasets/housing.csv'

st.sidebar.header(PROJECT_NAME)
st.sidebar.write(AUTHOR)

def load_data(rows = None):
    data = pd.read_csv(Datapath)
    return data

def load_model(path = 'models/iso_linear_model.pk'):
    with open(path, 'rb') as f:
        return pickle.load(f)

with st.spinner('loading anamoly dectection model'):
    model = load_model()
    isolationForest = load_model('models/isolation_model.pk')
    st.success('models loaded into memory')

dataHousing = load_data()

choice = st.sidebar.radio("Project Menu",MENU_OPTIONS)

if choice =='view data':
    st.title("View raw data")
    
    st.write('Dataset')
    st.write(dataHousing)

if choice =='view stats':
    st.title('View Statistics in Dataset')

    st.write('News Dataset')
    describeData = dataHousing.describe()
    st.write(describeData)

if choice =='visualize':
    st.title("Graphs and charts")

if choice =='prediction':
    st.title('Use AI to predict')
    st.subheader('fill the detail and get result')

    CRIM =st.number_input('per capita crime rate by town',min_value=0.0063, max_value=88.9762)
    ZN = st.number_input('proportion of residential land zoned for lots over 25,000 sq.ft.',min_value=0.0, max_value=100.0)
    INDUS = st.number_input('proportion of non-retail business acres per town',min_value=0.4600, max_value=27.7400)
    CHAS = st.number_input('Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)',min_value=0.0, max_value=1.0)
    NOX = st.number_input('nitric oxides concentration (parts per 10 million)',min_value=0.3850, max_value=0.8710)
    RM = st.number_input('average number of rooms per dwelling',min_value=3.5610, max_value=8.7800)
    AGE = st.number_input('proportion of owner-occupied units built prior to 1940',min_value=2.9000, max_value=100.0)
    DIS = st.number_input('weighted distances to five Boston employment centres',min_value=1.1296, max_value=12.1265)
    RAD = st.number_input('index of accessibility to radial highways',min_value=1, max_value=24)
    TAX = st.number_input('full-value property-tax rate per $10,000',min_value=187, max_value=711)
    PTRATIO = st.number_input('pupil-teacher ratio by town',min_value=12.6000, max_value=22.0)
    B = st.number_input('1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',min_value=0.3200, max_value=396.9000)
    LSTAT = st.number_input('percent lower status of the population',min_value=1.7300, max_value=37.9700)


    clicked = st.button("Make Anamoly Detection")
    if clicked and model:
        st.title('Anamoly Detection')

        features = np.array([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT])

        prediction = model.predict(features.reshape(1, -1))
        st.header("Anamoly detection")
        st.success(prediction[0])

if choice =='history':
    st.title('Previous prediction')

if choice =='about':
    st.title('About the project')
    st.image('img.png')
    st.write("""Most user management systems have some sort of main page, usually known as a dashboard. You’ll create a dashboard in this section, but because it won’t be the only page in your application, you’ll also create a base template to keep the looks of the website consistent.
You won’t use any of Django’s advanced template features, but if you need a refresher on the template syntax, then you might want to check out Django’s template documentation""")

if choice == 'upload':
    st.title("Upload image")
    img = st.file_uploader("Select image",type=['jpg','png','bmp'])
    st.write(img)
    if img:
        st.image(img.read())
    st.button('Make Predicton')
