import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from PIL import Image

# 1) 모델·스케일러·Explainer 로드
model60   = joblib.load('ua60_model')
scaler60  = joblib.load('scaler60.pkl')
explainer = shap.TreeExplainer(model60)

# 2) 참고 이미지 로드
im_non     = Image.open('non.jpg')
im_normal  = Image.open('normal.jpg')
im_abnormal= Image.open('abnormal.jpg')
im_pro     = Image.open('pro.jpg')

# — 예측 함수
def model_prediction(sample_case, scaler=scaler60, model=model60):
    std_cols = ['age', 'he_uph', 'he_usg']
    feats = sample_case.loc[:, ['male','he_usg','he_uph','he_ubld','he_uglu','he_upro','age']].copy()
    feats.loc[:, std_cols] = scaler.transform(feats[std_cols])
    prob = model.predict_proba(feats)[:,1]
    return np.float64(prob)

# — 입력 매핑 함수
def data_mapping(df):
    df = df.copy()
    df['male'] = df['male'].map({'female':0, 'male':1})
    m = {"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5}
    for c in ['he_ubld','he_upro','he_uglu']:
        df[c] = df[c].map(m)
    return df

# — SHAP 값 계산
def get_shap_values(sample_case):
    feats = sample_case.loc[:, ['male','he_usg','he_uph','he_ubld','he_uglu','he_upro','age']].copy()
    std_cols = ['age','he_uph','he_usg']
    feats.loc[:, std_cols] = scaler60.transform(feats[std_cols])
    shap_all = explainer.shap_values(feats)
    # xgboost 일 때 리스트로 반환되면 양성 클래스(1)만 취함
    arr = shap_all[1] if isinstance(shap_all, list) else shap_all
    vals = arr[0]
    return pd.DataFrame({
        'shap_value(probability)': vals
    }, index=[
        'sex',
        'urine specific gravity',
        'urine pH',
        'urine blood',
        'urine glucose',
        'urine protein',
        'age'
    ])

def main():
    st.title("Check Your Function of Kidney")
    st.markdown(
        """<div style="background-color:grey;padding:13px">
           <h1 style="color:black;text-align:center;">eGFR60 Classifier ML App</h1>
           </div>""",
        unsafe_allow_html=True
    )

    # — 사이드바 입력
    age    = st.sidebar.slider("Age", 0, 100, 30)
    sex    = st.sidebar.selectbox("Sex", ("female","male"))
    he_usg = st.sidebar.selectbox("Urine Specific Gravity",
                (1.000,1.005,1.010,1.015,1.020,1.025,1.030))
    he_uph = st.sidebar.selectbox("Urine pH",
                (5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0))
    he_ubld= st.sidebar.selectbox("Urine Blood", ("-","+/-","1+","2+","3+","4+"))
    he_uglu= st.sidebar.selectbox("Urine Glucose", ("-","+/-","1+","2+","3+","4+"))
    he_upro= st.sidebar.selectbox("Urine Protein", ("-","+/-","1+","2+","3+","4+"))

    raw = {
        'male': sex,
        'he_usg': he_usg,
        'he_uph': he_uph,
        'he_ubld': he_ubld,
        'he_uglu': he_uglu,
        'he_upro': he_upro,
        'age': age
    }
    df = pd.DataFrame(raw, index=[0])

    if st.button("Predict"):
        mapped   = data_mapping(df)
        prob     = model_prediction(mapped)
        shap_df  = get_shap_values(mapped)

        st.success(f'Probability: {prob:.2f}')

        # — 업로딩 단계에 따른 threshold 분기
        if mapped['he_upro'].item() <= 1:
            thresh = 0.44
        else:
            thresh = 0.77
        st.success(f"Threshold: {thresh}")

        # — 결과에 따른 이미지 출력
        if prob > thresh:
            st.image(im_abnormal, caption='Abnormal')
            if mapped['he_upro'].item() <= 1:
                st.image(im_non, caption='Reference')
            else:
                st.image(im_pro, caption='Reference')
        else:
            st.image(im_normal, caption='Normal')

        # — SHAP 바 차트
        st.bar_chart(shap_df)

if __name__ == '__main__':
    main()
