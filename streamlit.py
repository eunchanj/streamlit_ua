# streamlit test, version1

import pandas as pd
import numpy as np
from pickle import load
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from numpy import sqrt
from numpy import argmax
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from pickle import load
import shap

# loading in the model to predict on the data
## xgb models
model60 = joblib.load('ua60_model')
## scaler
scaler60 = joblib.load('scaler60.pkl')
## explainer
expl60 = joblib.load('explainer60.pkl')

## im non
im_non = Image.open('non.jpg')
## im pro
im_pro = Image.open('pro.jpg')
## im normal
im_normal = Image.open('normal.jpg')
## im normal
im_abnormal = Image.open('abnormal.jpg')

# custom def : shap
def shap(
    sample_case,
    scaler = scaler60,
    explainer = expl60
):
    # standardization columns
    std_cols=['age','he_uph','he_usg']    
    # feature extraction from input data UA 
    sample_case_features = sample_case.loc[:,['male', 'he_usg', 'he_uph', 'he_ubld', 'he_uglu', 'he_upro', 'age']]
    sample_case_features[std_cols] = scaler.transform(sample_case_features[std_cols])
    expl_test = expl60.shap_values(sample_case_features.iloc[0])
    shap_bar = pd.DataFrame(
        {'shap_value(probability)'  : expl_test}, index =  ['sex', 'urine specific gravity', 'urine pH', 'urine blood', 'urine glucose', 'urien protein', 'age'])
  #  clrs = ['blue' if x < 0 else 'red' for x in shap_var['shap']]
    return shap_bar

# custom def : standardization and prediction
def model_prediction(
    sample_case,
    scaler = joblib.load('scaler60.pkl'),
    model = joblib.load('ua60_model')
):
    """
    UA5 type model
    he_usg = Urine specific gravity
    he_uph = Urine pH
    he_ubld = Urine blood
    he_uglu = Urine glucose
    he_upro = Urine protein
    """
    
    # standardization columns
    std_cols=['age','he_uph','he_usg']    
    # feature extraction from input data UA 
    sample_case_features = sample_case.loc[:,['male', 'he_usg', 'he_uph', 'he_ubld', 'he_uglu', 'he_upro', 'age']]
    sample_case_features[std_cols] = scaler.transform(sample_case_features[std_cols])
    
    # predict probability by model
    prob = model.predict_proba(sample_case_features)[:,1]
            
    return prob

def data_mapping(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.male = df.male.map({'female':0, 'male':1})
    df.he_ubld = df.he_ubld.map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    df.he_upro = df.he_upro.map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    df.he_uglu = df.he_uglu.map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    return df

def main():
      # giving the webpage a title
    st.title("check your function of kidney")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:grey;padding:13px">
    <h1 style ="color:black;text-align:center;">eGFR60 classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction

    age = st.sidebar.slider("age", 0, 100, 1)
    male = st.sidebar.selectbox("sex", ("female", "male"))    
    he_usg = st.sidebar.selectbox("urine specific gravity", (1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030))
    he_uph = st.sidebar.selectbox("urine pH", (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0))
    he_ubld = st.sidebar.selectbox("urine blood", ("-", "+/-", "1+", "2+", "3+", "4+"))
    he_uglu = st.sidebar.selectbox("urine glucose", ("-", "+/-", "1+", "2+", "3+", "4+"))
    he_upro = st.sidebar.selectbox("urine protein", ("-", "+/-", "1+", "2+", "3+", "4+"))
    
    features = {"male"    : male,
               "he_usg"   : he_usg,
               "he_uph"   : he_uph,
               "he_ubld"  : he_ubld,
               "he_uglu"  : he_uglu,
               "he_upro"  : he_upro,
               "age"      : age}
    sample_case = pd.DataFrame(features, index=[0])
    
    result = ""
    prob = 0.0
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        sample_case_map = data_mapping(sample_case)
        result = model_prediction(sample_case_map)
        prob = result
        shap_bar = shap(sample_case_map)

        st.success('probability : {}'.format(result))
    
        if ((sample_case_map['he_upro'].item()<=1) and (prob > 0.44) :
            #st.success("prediction : eGFR<60, abnormal")
            st.success("threshold : 0.44")
            st.image(im_abnormal)
            st.image(im_non, caption='reference')
            st.bar_chart(data=shap_bar)
        elif ((sample_case_map['he_upro'].item()<=1) and (prob <= 0.44) :
            #st.success("prediction : eGFR>=60, normal")
            st.success("threshold : 0.44")
            st.image(im_normal)
            st.bar_chart(data=shap_bar)
        elif((sample_case_map['he_upro'].item()>1) and (prob > 0.77) :
            #st.success("prediction : eGFR<60, abnormal")
            st.success("threshold : 0.77")
            st.image(im_abnormal)
            st.bar_chart(data=shap_bar)
            st.image(im_pro, caption='reference')
        else :
            #st.success("prediction : eGFR>=60, normal")
            st.success("threshold : 0.77")
            st.image(im_normal)
            st.bar_chart(data=shap_bar)
     
if __name__=='__main__':
    main()

