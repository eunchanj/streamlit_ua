import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# 1) 모델과 스케일러 로드
model60  = joblib.load('ua60_model')
scaler60 = joblib.load('scaler60.pkl')

# 2) Explainer 를 런타임에 직접 생성
expl60 = shap.TreeExplainer(model60)

# 이미지 로딩
im_non      = Image.open('non.jpg')
im_pro      = Image.open('pro.jpg')
im_normal   = Image.open('normal.jpg')
im_abnormal = Image.open('abnormal.jpg')

# SHAP 값 계산 함수
def get_shap_values(sample_case, scaler=scaler60, explainer=expl60):
    std_cols = ['age', 'he_uph', 'he_usg']
    feats = sample_case[['male','he_usg','he_uph','he_ubld','he_uglu','he_upro','age']]
    feats[std_cols] = scaler.transform(feats[std_cols])
    shap_vals = explainer.shap_values(feats.iloc[0])
    return pd.DataFrame(
        {'shap_value(probability)': shap_vals},
        index=['sex','urine specific gravity','urine pH','urine blood','urine glucose','urien protein','age']
    )

# 예측 함수
def model_prediction(sample_case, scaler=scaler60, model=model60):
    std_cols = ['age', 'he_uph', 'he_usg']
    feats = sample_case[['male','he_usg','he_uph','he_ubld','he_uglu','he_upro','age']]
    feats[std_cols] = scaler.transform(feats[std_cols])
    return np.float64(model.predict_proba(feats)[:,1])

# 입력 매핑
def data_mapping(df):
    df.male = df.male.map({'female':0, 'male':1})
    for col in ['he_ubld','he_upro','he_uglu']:
        df[col] = df[col].map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    return df

def main():
    st.title("Check Your Function of Kidney")
    st.markdown(
        """
        <div style="background-color:grey;padding:13px">
        <h1 style="color:black;text-align:center;">eGFR60 Classifier ML App</h1>
        </div>
        """, unsafe_allow_html=True
    )

    age     = st.sidebar.slider("Age", 0, 100, 1)
    male    = st.sidebar.selectbox("Sex", ("female","male"))
    he_usg  = st.sidebar.selectbox("Urine Specific Gravity", (1.000,1.005,1.010,1.015,1.020,1.025,1.030))
    he_uph  = st.sidebar.selectbox("Urine pH", (5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0))
    he_ubld = st.sidebar.selectbox("Urine Blood", ("-","+/-","1+","2+","3+","4+"))
    he_uglu = st.sidebar.selectbox("Urine Glucose", ("-","+/-","1+","2+","3+","4+"))
    he_upro = st.sidebar.selectbox("Urine Protein", ("-","+/-","1+","2+","3+","4+"))

    sample_case = pd.DataFrame({
        "male":    [male],
        "he_usg":  [he_usg],
        "he_uph":  [he_uph],
        "he_ubld": [he_ubld],
        "he_uglu": [he_uglu],
        "he_upro": [he_upro],
        "age":     [age]
    })

    if st.button("Predict"):
        mapped = data_mapping(sample_case)
        prob   = model_prediction(mapped)
        shap_df= get_shap_values(mapped)

        st.success(f'Probability: {prob:.2f}')

        # 결과 표시
        if mapped['he_upro'].item() <= 1:
            thresh = 0.44
            img    = im_non if prob <= thresh else im_abnormal
            st.success(f"Threshold: {thresh}")
            st.image(img)
            if prob > thresh:
                st.image(im_non, caption='Reference')
        else:
            thresh = 0.77
            img    = im_normal if prob <= thresh else im_abnormal
            st.success(f"Threshold: {thresh}")
            st.image(img)
            if prob > thresh:
                st.image(im_pro, caption='Reference')

        st.bar_chart(shap_df)

if __name__ == '__main__':
    main()
