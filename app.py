import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title('🔮 Previsão de Churn')
st.write('Preencha os dados do cliente para prever a probabilidade de churn.')

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox('É idoso?', ['Não', 'Sim'])
    partner = st.selectbox('Tem parceiro(a)?', ['Yes', 'No'])
    dependents = st.selectbox('Tem dependentes?', ['Yes', 'No'])
    tenure = st.number_input('Tempo de contrato (meses)', min_value=0, max_value=72)
    monthly_charges = st.number_input('Cobrança mensal', min_value=0.0, max_value=200.0)
    total_charges = st.number_input('Cobrança total', min_value=0.0, max_value=10000.0)
    multiple_lines = st.selectbox('Múltiplas linhas', ['No', 'No phone service', 'Yes'])
    internet_service = st.selectbox('Serviço de internet', ['DSL', 'Fiber optic', 'No'])

with col2:
    online_security = st.selectbox('Segurança online', ['No', 'No internet service', 'Yes'])
    online_backup = st.selectbox('Backup online', ['No', 'No internet service', 'Yes'])
    tech_support = st.selectbox('Suporte técnico', ['No', 'No internet service', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'No internet service', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'No internet service', 'Yes'])
    contract = st.selectbox('Tipo de contrato', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Fatura digital', ['No', 'Yes'])
    payment_method = st.selectbox('Método de pagamento', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
    device_protection = st.selectbox('Proteção do dispositivo', ['No', 'No internet service', 'Yes'])

if st.button('Prever Churn'):
    # Monta o dicionário com todos os valores zerados
    input_dict = {col: 0 for col in [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes'
    ]}

    # Preenche os valores
    input_dict['SeniorCitizen'] = 1 if senior_citizen == 'Sim' else 0
    input_dict['Partner'] = 1 if partner == 'Yes' else 0
    input_dict['Dependents'] = 1 if dependents == 'Yes' else 0
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict[f'MultipleLines_{multiple_lines}'] = 1
    input_dict[f'InternetService_{internet_service}'] = 1
    input_dict[f'OnlineSecurity_{online_security}'] = 1
    input_dict[f'OnlineBackup_{online_backup}'] = 1
    input_dict[f'TechSupport_{tech_support}'] = 1
    input_dict[f'StreamingTV_{streaming_tv}'] = 1
    input_dict[f'StreamingMovies_{streaming_movies}'] = 1
    input_dict[f'Contract_{contract}'] = 1
    input_dict[f'PaperlessBilling_{paperless_billing}'] = 1
    input_dict[f'PaymentMethod_{payment_method}'] = 1
    input_dict[f'DeviceProtection_{device_protection}'] = 1

    # Transforma em DataFrame e escala
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Previsão
    proba = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.divider()

    if pred == 1:
        st.error(f'⚠️ Alto risco de churn! Probabilidade: {proba:.1%}')
    else:
        st.success(f'✅ Baixo risco de churn! Probabilidade: {proba:.1%}')