import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. 载入模型
model_xgb = joblib.load('xgb.pkl')

# 2. 配置 SHAP 解释器
feature_label = ['IC_DeepFeature1147', 'Zeff_DeepFeature1540', 'VMI_exponential_glrlm_RunVariance', 
                 'Zeff_wavelet-LHL_glszm_LargeAreaHighGrayLevelEmphasis', 'IC_gradient_firstorder_Kurtosis', 
                 'VMI_exponential_firstorder_Kurtosis', 'IC_DeepFeature288', 'Zeff_DeepFeature309', 
                 'VMI_DeepFeature1645', 'IC_DeepFeature599', 'Zeff_DeepFeature928', 'VMI_DeepFeature1109', 
                 'PEI_DeepFeature1021', 'VMI_DeepFeature1817', 'VMI_logarithm_glszm_SmallAreaLowGrayLevelEmphasis', 
                 'IC_DeepFeature1877', 'PEI_DeepFeature1218', 'Zeff_wavelet-HHL_gldm_DependenceVariance', 
                 'IC_original_glszm_GrayLevelNonUniformity', 'Zeff_wavelet-HLL_glszm_SizeZoneNonUniformityNormalized', 
                 'IC_wavelet-LLH_ngtdm_Coarseness', 'Zeff_DeepFeature763', 'IC_DeepFeature1130', 'IC_DeepFeature1791']

# 3. Streamlit 输入
st.title('XGBoost Predictive Model and SHAP Analysis')
st.sidebar.header('输入特征')

# 输入特征表单
inputs = {}
for feature in feature_label:
    inputs[feature] = st.sidebar.number_input(feature, min_value=-10.0, max_value=10.0, value=0.0)

# 将输入的特征值转换为 Pandas DataFrame
input_df = pd.DataFrame([inputs])

# 4. 预测按钮
if st.sidebar.button('预测'):
    # 预测
    dinput = xgb.DMatrix(input_df.values)
    prediction = model_xgb.predict(dinput)[0]

    # 展示预测结果
    st.subheader('预测结果')
    st.write(f'预测值: {prediction}')

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(input_df)

    # 5. 显示 SHAP 力图
    st.subheader('SHAP 力图')
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0, :], feature_names=feature_label)

    # 可选：保存 SHAP 力图为图片（可选步骤）
    # plt.savefig('force_plot_sample0.png', dpi=300, bbox_inches="tight", pad_inches=0.1)
    # st.image('force_plot_sample0.png')

