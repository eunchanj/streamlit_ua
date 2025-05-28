# 중복 import 제거 및 함수명 변경
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
from PIL import Image
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
# from sklearn.metrics import plot_roc_curve  # Deprecated in recent versions

# 로딩된 모델 및 scaler
model60 = joblib.load('ua60_model')
scaler60 = joblib.load('scaler60.pkl')
expl60 = joblib.load('explainer60.pkl')
#expl60 = shap.TreeExplainer(model60)

# 이미지 로딩
im_non = Image.open('non.jpg')
im_pro = Image.open('pro.jpg')
im_normal = Image.open('normal.jpg')
im_abnormal = Image.open('abnormal.jpg')

# SHAP 시각화용 데이터프레임 생성 함수 (함수명 충돌 방지 위해 이름 변경)
def get_shap_values(sample_case, scaler=scaler60, explainer=expl60):
    std_cols = ['age', 'he_uph', 'he_usg']
    sample_case_features = sample_case.loc[:, ['male', 'he_usg', 'he_uph', 'he_ubld', 'he_uglu', 'he_upro', 'age']]
    sample_case_features[std_cols] = scaler.transform(sample_case_features[std_cols])
    shap_values = explainer.shap_values(sample_case_features.iloc[0])
    shap_bar = pd.DataFrame(
        {'shap_value(probability)': shap_values},
        index=['sex', 'urine specific gravity', 'urine pH', 'urine blood', 'urine glucose', 'urien protein', 'age']
    )
    return shap_bar

# 모델 예측 함수
def model_prediction(sample_case, scaler=scaler60, model=model60):
    std_cols = ['age', 'he_uph', 'he_usg']
    sample_case_features = sample_case.loc[:, ['male', 'he_usg', 'he_uph', 'he_ubld', 'he_uglu', 'he_upro', 'age']]
    sample_case_features[std_cols] = scaler.transform(sample_case_features[std_cols])
    prob = model.predict_proba(sample_case_features)[:, 1]
    return np.float64(prob)

# 입력값 매핑 함수
def data_mapping(df):
    df.male = df.male.map({'female': 0, 'male': 1})
    for col in ['he_ubld', 'he_upro', 'he_uglu']:
        df[col] = df[col].map({"-": 0, "+/-": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5})
    return df

# Streamlit 메인 함수
def main():
    st.title("Check Your Function of Kidney")

    st.markdown(
        """
        <div style="background-color:grey;padding:13px">
        <h1 style="color:black;text-align:center;">eGFR60 Classifier ML App</h1>
        </div>
        """, unsafe_allow_html=True)

    age = st.sidebar.slider("Age", 0, 100, 1)
    male = st.sidebar.selectbox("Sex", ("female", "male"))
    he_usg = st.sidebar.selectbox("Urine Specific Gravity", (1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030))
    he_uph = st.sidebar.selectbox("Urine pH", (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0))
    he_ubld = st.sidebar.selectbox("Urine Blood", ("-", "+/-", "1+", "2+", "3+", "4+"))
    he_uglu = st.sidebar.selectbox("Urine Glucose", ("-", "+/-", "1+", "2+", "3+", "4+"))
    he_upro = st.sidebar.selectbox("Urine Protein", ("-", "+/-", "1+", "2+", "3+", "4+"))

    features = {
        "male": male,
        "he_usg": he_usg,
        "he_uph": he_uph,
        "he_ubld": he_ubld,
        "he_uglu": he_uglu,
        "he_upro": he_upro,
        "age": age
    }
    sample_case = pd.DataFrame(features, index=[0])

    if st.button("Predict"):
        sample_case_mapped = data_mapping(sample_case)
        prob = model_prediction(sample_case_mapped)
        shap_bar = get_shap_values(sample_case_mapped)

        st.success(f'Probability: {prob:.2f}')

        if ((sample_case_mapped['he_upro'].item() <= 1) and (prob > 0.44)):
            st.success("Threshold: 0.44")
            st.image(im_abnormal)
            st.image(im_non, caption='Reference')
            st.bar_chart(shap_bar)
        elif ((sample_case_mapped['he_upro'].item() <= 1) and (prob <= 0.44)):
            st.success("Threshold: 0.44")
            st.image(im_normal)
            st.bar_chart(shap_bar)
        elif ((sample_case_mapped['he_upro'].item() > 1) and (prob > 0.77)):
            st.success("Threshold: 0.77")
            st.image(im_abnormal)
            st.bar_chart(shap_bar)
            st.image(im_pro, caption='Reference')
        else:
            st.success("Threshold: 0.77")
            st.image(im_normal)
            st.bar_chart(shap_bar)

if __name__ == '__main__':
    main()
