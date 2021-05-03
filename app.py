import streamlit as st
from config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import webcam

Datapath = 'datasets/mice.csv'

st.sidebar.header(PROJECT_NAME)
st.sidebar.write(AUTHOR)

def load_data(rows = None):
    data = pd.read_csv(Datapath)
    return data

dataMice = load_data()

choice = st.sidebar.radio("Project Menu",MENU_OPTIONS)

if choice =='view data':
    st.title("View raw data")
    
    st.write('Dataset')
    st.write(dataMice)

if choice =='view stats':
    st.title('View Statistics in Dataset')

    st.write('News Dataset')
    describeData = dataMice.describe()
    st.write(describeData)

if choice =='visualize':
    st.title("Graphs and charts")

if choice =='prediction':
    st.title('Use AI to predict')
    st.subheader('fill the detail and get result')
    customerID = st.text_input("enter customer id")
    state = st.text_input('enter state')
    has_card = st.checkbox('do you have credit card')
    credit_score = st.number_input('credit score', min_value=0, max_value=100)
    clicked = st.button("Click to classify customer")
    if clicked:
        st.success("abhi bahut code baki h")

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

if choice == 'webcam':
    st.title("use webcam for realtime prediction")
    st.info("Please follow the code of learning")
    cam_num = st.number_input('select camera',min_value=1,max_value=5,value=1)
    clicked = st.button('Launch camera')
    if clicked:
        webcam.camera(cam_num-1)