# -*- coding: utf-8 -*-
"""
Created on Sat May  1 03:15:51 2021

@author: Rahul
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly_express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
sc=StandardScaler()


global df
df=pd.read_csv('PlacementData.csv')
global numeric_columns
global non_numeric_columns
def welcome():
    return "Welcome All"



pickle_in1= open("placed.pkl","rb")
pickle_in2=open("salary.pkl","rb")
placed=pickle.load(pickle_in1)
salary=pickle.load(pickle_in2)

# Predict Salary
def func2(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,special,mba_p,status):
    if(gender=='M'):
        gen=1
    if(gender=='F'):
        gen=0
    if(ssc_b=='Others') :
        ssc_board=1
    if(ssc_b=='Central'):
        ssc_board=0
    if(hsc_b=='Others'):
        hsc_board=1
    if(hsc_b=='Central'):
        hsc_board=0
    if(workex=='No'):
        work=0
    if(workex=='Yes'):
        work=1
    if(special=='Mkt&Fin'):
        spec=0
    if(special=='Mkt&HR'):
        spec=1
    if(status=='Placed'):
        stat=1
    if(status=='Not Placed'):
        stat=0
    hsc_s1=0
    hsc_s2=0
    hsc_s3=0
    if(hsc_s=='Commerce'):
        hsc_s2=1
    if(hsc_s=='Science'):
        hsc_s3=1
    if(hsc_s=='Arts'):
        hsc_s1=1
    degree_t1=0
    degree_t2=0
    degree_t3=0
    if(degree_t=='Sci&Tech'):
        degree_t3=1
    if(degree_t=='Comm&Mgmt'):
        degree_t1=1
    if(degree_t=='Others'):
        degree_t2=1
        
    X = [[gen,ssc_p,ssc_board,hsc_p,hsc_board,degree_p,work,etest_p,spec,mba_p,stat,hsc_s1,hsc_s2,hsc_s3,degree_t1,degree_t2,degree_t3]]

    X = pd.DataFrame(X)
    y_pred = salary.predict(X)
    return y_pred

# Predict Placement
def func1(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,special,mba_p):
    if(gender=='M'):
        gen=1
    if(gender=='F'):
        gen=0
    if(ssc_b=='Others') :
        ssc_board=1
    if(ssc_b=='Central'):
        ssc_board=0
    if(hsc_b=='Others'):
        hsc_board=1
    if(hsc_b=='Central'):
        hsc_board=0
    if(workex=='No'):
        work=0
    if(workex=='Yes'):
        work=1
    if(special=='Mkt&Fin'):
        spec=0
    if(special=='Mkt&HR'):
        spec=1
    hsc_s1=0
    hsc_s2=0
    hsc_s3=0
    if(hsc_s=='Commerce'):
        hsc_s2=1
    if(hsc_s=='Science'):
        hsc_s3=1
    if(hsc_s=='Arts'):
        hsc_s1=1
    degree_t1=0
    degree_t2=0
    degree_t3=0
    if(degree_t=='Sci&Tech'):
        degree_t3=1
    if(degree_t=='Comm&Mgmt'):
        degree_t1=1
    if(degree_t=='Others'):
        degree_t2=1
    
    X = [[gen,ssc_p,ssc_board,hsc_p,hsc_board,degree_p,work,etest_p,spec,mba_p,hsc_s1,hsc_s2,hsc_s3,degree_t1,degree_t2,degree_t3]]

    X = pd.DataFrame(X)
    y_pred = placed.predict(X)
    return y_pred

print(func2('F',94.2,'Central',90.0,'Others','Science',95.3,'Sci&Tech','Yes',85,'Mkt&HR',98.0,'Placed'))
print(func1('F',94.2,'Central',90.0,'Others','Science',95.3,'Sci&Tech','Yes',85,'Mkt&HR',98.0))


def main():
    st.title("Predict Salary")
    gender = st.text_input("Gender: M/F")
    ssc_p = float(st.number_input("SSC percentile"))
    ssc_b = st.text_input("SSC B:Central/Others")
    hsc_p = float(st.number_input("HSC percentile"))
    hsc_b = st.text_input("HSC B: Central/Others")
    hsc_s = st.text_input("HSC S: Commerce/Science/Arts")
    degree_p = float(st.number_input("Degree percentile"))
    degree_t = st.text_input("Degree T: Sci&Tech/Comm&Mgmt/Others")
    workex = st.text_input("Workex: Yes/No")
    etest_p =  float(st.number_input("E-test percentile"))
    special = st.text_input("Special: Mkt&Fin/Mkt&HR")
    mba_p = float(st.number_input("MBA percentile"))
    status = st.text_input("Status: Placed/Not Placed")

    if st.button("Predict Salary"):
        output = func2(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,special,mba_p,status)
        st.success('Estimated salary: {}'.format(output))
    if st.button("Placed or Not"):
        output = func1(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,special,mba_p)
        st.success('Prediction {}'.format(output))
    agree=st.checkbox("Hide Graphs")
    if (not agree):
        st.sidebar.subheader("Visualization Settings")
        try:
            st.write(df)
            numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
           
        
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
        
            non_numeric_columns.append(None)
            print(numeric_columns)
        except Exception as e:
            print(e)
            st.write("Please upload file to the application.")
        # add a select widget to the side bar
        chart_select = st.sidebar.selectbox(
            label="Select the chart type",
            options=['Histogram', 'Lineplots','Scatterplots',  'Boxplot'])

        if chart_select == 'Scatterplots':
            st.sidebar.subheader("Scatterplot Settings")
            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
                # display the chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
    
        if chart_select == 'Lineplots':
            st.sidebar.subheader("Line Plot Settings")
            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
    
        if chart_select == 'Histogram':
            st.sidebar.subheader("Histogram Settings")
            try:
                x = st.sidebar.selectbox('Feature', options=numeric_columns)
                bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                         max_value=100, value=40)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.histogram(x=x, data_frame=df, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                    print(e)
    
        if chart_select == 'Boxplot':
            st.sidebar.subheader("Boxplot Settings")
            try:
                y = st.sidebar.selectbox("Y axis", options=numeric_columns)
                x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.box(data_frame=df, y=y, x=x, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
            
        
   

if __name__=='__main__':
    main()

