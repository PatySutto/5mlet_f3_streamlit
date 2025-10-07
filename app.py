import streamlit as st
import numpy as np
import joblib


st.title("❤️ Classificação Doença Cardiovascular")
st.write("Preencha os campos abaixo para obter a previsão:")

# Entradas numéricas
max_hr = st.number_input("Frequência cardíaca máxima", min_value=60, max_value=220, value=150, step=1)
oldpeak = st.number_input("Depressão do ST (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.5)

# Entradas categóricas
# ------------------------------------------------------------
chest_pain_type = st.selectbox(
    "Tipo de Dor no Peito (ChestPainType)",
    ["ASY", "ATA", "NAP", "TA"]
)
# Mapeando para números
chest_pain_type_map = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
chest_pain_type_val = chest_pain_type_map[chest_pain_type]

exercise_angina = st.selectbox(
    "Angina ao Exercício (ExerciseAngina)",
    ["N", "Y"]
)
exercise_angina_map = {"N": 0, "Y": 1}
exercise_angina_val = exercise_angina_map[exercise_angina]

st_slope = st.selectbox(
    "Inclinação do ST (ST_Slope)",
    ["Down", "Flat", "Up"]
)
st_slope_map = {"Down": 0, "Flat": 1, "Up": 2}
st_slope_val = st_slope_map[st_slope]


# Agrupar dados do usuário
# ------------------------------------------------------------
# Ordem da entrada: ['ExerciseAngina' 'Oldpeak' 'ST_Slope' 'MaxHR' 'ChestPainType']
dados_usuario = np.array([[exercise_angina_val, oldpeak, st_slope_val, max_hr, 
                           chest_pain_type_val]])


# Botão de previsão
# ------------------------------------------------------------
if st.button("🔍 Fazer previsão"):
    try:
        # Carregar o modelo salvo
        modelo = joblib.load('modelo_regressao_logistica.joblib')

        # Fazer previsão
        predicao = modelo.predict(dados_usuario)
        probabilidade = modelo.predict_proba(dados_usuario)[0][1]

        print(predicao)
        print(f"Probabilidade de classe positiva: {probabilidade * 100:.2f}%")
        print(dados_usuario)

        # Mostrar resultado
        if predicao[0] == 1:
            st.success(f"✅ Resultado **Positivo** para doença cardiovascular (prob: {probabilidade*100:.2f}%)")
        else:
            st.info(f"❌ Resultado **Negativo** para doença cardiovascular (prob: {(1-probabilidade)*100:.2f}%)")

    except FileNotFoundError:
        st.error("❌ Arquivo 'modelo_regressao_logistica.joblib' não encontrado!")
    except Exception as e:
        st.error(f"⚠️ Erro ao executar o modelo: {e}")

st.caption("Feito com Streamlit")
