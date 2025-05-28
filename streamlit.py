import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from PIL import Image

# 1) 모델과 스케일러 로드
model60  = joblib.load('ua60_model')
scaler60 = joblib.load('scaler60.pkl')

# 2) SHAP Explainer 런타임에 생성
explainer60 = shap.TreeExplainer(model60)

# 원본 입력값을 모델 입력 포맷으로 매핑하는 함수
def data_mapping(raw):
    return pd.DataFrame([{
        'male':        1 if raw['sex']=='Male' else 0,
        'he_usg':      raw['urine specific gravity'],
        'he_uph':      raw['urine pH'],
        'he_ubld':     raw['urine blood'],
        'he_uglu':     raw['urine glucose'],
        'he_upro':     raw['urine protein'],
        'age':         raw['age']
    }])

# SHAP 값 계산
def get_shap_values(sample_case):
    feats = sample_case[['male','he_usg','he_uph','he_ubld','he_uglu','he_upro','age']].copy()
    std_cols = ['age','he_uph','he_usg']
    # .values 로 넘겨서 feature name 경고 제거
    feats.loc[:, std_cols] = scaler60.transform(feats[std_cols].values)
    # explainer.shap_values에 2D DataFrame(1×7) 그대로 넘김
    shap_all = explainer60.shap_values(feats)
    # 양성 클래스(1)의 첫 샘플 shap 값만 취함
    shap_pos = shap_all[1][0]
    return pd.DataFrame(
        {'shap_value(probability)': shap_pos},
        index=['sex',
               'urine specific gravity',
               'urine pH',
               'urine blood',
               'urine glucose',
               'urine protein',
               'age']
    )

# 예측 확률 계산
def model_prediction(sample_case):
    feats = sample_case[['male','he_usg','he_uph','he_ubld','he_uglu','he_upro','age']].copy()
    std_cols = ['age','he_uph','he_usg']
    feats.loc[:, std_cols] = scaler60.transform(feats[std_cols].values)
    return model60.predict_proba(feats)[:,1]

def main():
    st.title("Check Your Kidney Function")

    # 사이드바에서 사용자 입력 받기
    st.sidebar.header("Patient Information")
    age = st.sidebar.slider("Age", 1, 100, 30)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    usg = st.sidebar.slider("Urine Specific Gravity", 1.005, 1.030, 1.015, step=0.001)
    uph = st.sidebar.slider("Urine pH", 4.5, 8.0, 7.0)
    ubld = st.sidebar.slider("Urine Blood", 0, 5, 0)
    ugu = st.sidebar.slider("Urine Glucose", 0, 4, 0)
    upro = st.sidebar.slider("Urine Protein", 0, 2, 0)

    raw_input = {
        'age': age,
        'sex': sex,
        'urine specific gravity': usg,
        'urine pH': uph,
        'urine blood': ubld,
        'urine glucose': ugu,
        'urine protein': upro
    }
    sample_case = data_mapping(raw_input)

    if st.button("Predict"):
        prob    = model_prediction(sample_case)[0]
        shap_df = get_shap_values(sample_case)

        st.success(f'Predicted probability of risk: {prob:.2f}')

        # 결과에 따라 이미지 출력
        img = Image.open('non.jpg' if prob < 0.5 else 'low.jpg')
        st.image(img, use_column_width=True)

        # SHAP 차트
        st.bar_chart(shap_df)

if __name__ == '__main__':
    main()
