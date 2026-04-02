# 📡 Telecom Churn Prediction

Modelo de machine learning para prever quais clientes de uma empresa de telecomunicações têm maior probabilidade de cancelar o serviço (churn), permitindo ações proativas de retenção.

---

## 🎯 Objetivo

Empresas de telecom perdem receita significativa com churn. Identificar clientes em risco **antes** que cancelem permite intervenções direcionadas — como ofertas personalizadas ou melhorias no atendimento — reduzindo o custo de aquisição de novos clientes.

---

## 📦 Dataset

- **Fonte:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Tamanho:** 7.043 clientes, 21 variáveis
- **Target:** `Churn` — se o cliente cancelou ou não (classe desbalanceada: ~26% churn)

---

## 🗂️ Estrutura do Projeto

```
churn-prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset original
│   └── telco_churn_limpo.csv                   # dataset após limpeza
│
├── models/
│   ├── churn_model.pkl                         # modelo treinado
│   └── scaler.pkl                              # scaler
│
├── churn_eda.ipynb                             # análise exploratória
├── churn_ml.ipynb                              # modelagem e avaliação
├── app.py                                      # interface Streamlit
└── requirements.txt
```

---

## 🔍 Análise Exploratória (EDA)

- Distribuição das variáveis categóricas e numéricas
- Análise de correlação com o target via heatmap
- Boxplots de `tenure`, `MonthlyCharges` e `TotalCharges`
- Gráficos de frequência de cada variável vs Churn

**Principais insights:**
- Clientes com contrato **mês a mês** têm taxa de churn muito maior que contratos anuais
- **Tenure baixo** é forte indicador de churn — a maioria dos cancelamentos ocorre nos primeiros meses
- Clientes com **Fiber optic** churnam mais, possivelmente por insatisfação com o serviço
- Quanto maior a **fatura mensal**, maior o risco de churn

---

## 🤖 Modelagem

### Pré-processamento
- Tratamento de valores nulos e espaços em branco
- Encoding de variáveis categóricas com `pd.get_dummies`
- Remoção de features com baixa correlação com o target
- Normalização com `StandardScaler`
- Divisão treino/teste: 80/20 com `stratify=y`

### Modelos Treinados e Comparação

| Modelo | Acurácia | Recall Churn | Churns Perdidos |
|---|---|---|---|
| Regressão Logística | 80% | 58% | 158 |
| **Regressão Logística (balanced)** | **72%** | **79%** | **77** |
| XGBoost | 75% | 68% | 119 |
| Random Forest (balanced) | 78% | 49% | 191 |

### Modelo Final
**Regressão Logística com `class_weight='balanced'`**

O dataset é desbalanceado (~26% churn), então a métrica principal é o **Recall da classe positiva** — minimizar clientes em risco que o modelo deixa escapar. O custo de perder um cliente real é muito maior do que oferecer um desconto desnecessário.

```
              precision    recall  f1-score
           0       0.90      0.70      0.79
           1       0.49      0.79      0.61
    accuracy                           0.73
```

---

## 🚀 Interface — Streamlit

A aplicação permite inserir os dados de um cliente e retorna a probabilidade de churn em tempo real.

**Como rodar:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tecnologias

- Python 3.11
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit
- Joblib
