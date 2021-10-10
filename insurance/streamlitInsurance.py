import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt 
from plotly import graph_objects as go 


#streamlit run streamlitInsurance.py
#cd C:\Users\perei\Data science\dlpoy_projects\insurance


# file uploader
def getData(data):
    df = pd.read_csv(data)
    return(df)

def TargetDependent(data, col):
    allcols = list(data.columns)
    ind_feat = [x for x in allcols if x!=col]
    X = data[ind_feat]
    selected_feat=['incident_severity','insured_hobbies','policy_number',
                   'policy_annual_premium','insured_education_level',
                   'insured_zip','bindYear','capital-loss','auto_model',
                   'bodily_injuries','umbrella_limit','property_damage',
                   'age','inciDay','incident_state']
    X = X[selected_feat]
    Y = data[col]
    return (X, Y)

def clfmodel(clf, X, Y, seed, shuffleState, classBalance):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=seed, shuffle=shuffleState)
    if (classBalance==True):
        sm = SMOTE(random_state=seed)
        xtrain, ytrain = sm.fit_resample(xtrain, ytrain)
    clf.fit(xtrain, ytrain)
    pred = clf.predict(xtest)
    acc = accuracy_score(ytest, pred)
    return(acc)

def model(name, X, Y, seed, shuffleState, classBalance):
    if (name=="Logistic regression"):
        acc = clfmodel(LogisticRegression(), X, Y, seed, shuffleState, classBalance)
    elif (name=="KNN classification"):
        acc = clfmodel(KNeighborsClassifier(), X, Y, seed, shuffleState, classBalance)
    elif (name=="SVM"):
        acc = clfmodel(SVC(), X, Y, seed, shuffleState, classBalance)
    elif (name=="Random forest"):
        acc = clfmodel(RandomForestClassifier(), X, Y, seed, shuffleState, classBalance)
    elif (name=="Light GBM"):
        acc = clfmodel(LGBMClassifier(), X, Y, seed, shuffleState, classBalance)
    elif (name=="XG boost"):
        acc = clfmodel(XGBClassifier(), X, Y, seed, shuffleState, classBalance)
    else:  
        acc="no model"
    return (acc)


def modeling(df, feat_target, algo, shuffleBool, randomState, classBalance):
    x, y = TargetDependent(df, feat_target)
    acc = model(algo, x, y, randomState, shuffleBool, classBalance)
    if (acc=="no model"):
        st.subheader("Warning: ")
        st.warning("Select any Machine learning model")
    else:
        st.write("Accuracy: ",acc)
        

# App layout
st.image("fraud.png")
fl_csv=st.file_uploader("Choose a CSV file")
if fl_csv is not None:
    df = getData(fl_csv)
    st.sidebar.write("Machine Learning Algorithm")
    algo=st.sidebar.selectbox("Choose the algorithm", ("Select", "Logistic regression", "KNN classification", "SVM", 
                                          "Random forest", "Light GBM", "XG boost"))
    shuffleBool=st.sidebar.checkbox("Check if shuffle is required")
    classBalance = st.sidebar.checkbox("Check if class balance is required")
    randomState = st.sidebar.slider("Choose the random state", min_value=0, max_value=100)
    feat_target=st.text_input("Choose the target feature")
    if (feat_target in df.columns):
        modeling(df, feat_target, algo, shuffleBool, randomState, classBalance)
