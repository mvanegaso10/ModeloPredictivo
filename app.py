
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('modelo_desercion.pkl')

st.set_page_config(page_title="Predicción de Deserción Estudiantil", page_icon="🎓")

st.title("🎓 Predicción de Deserción Estudiantil")
st.write("Ingrese los datos del estudiante para predecir la probabilidad de deserción.")

# --- Widgets para la entrada de datos ---

st.sidebar.header('Parámetros del Estudiante')

edad = st.sidebar.slider('Edad', min_value=18, max_value=25, value=20, step=1)
promedio = st.sidebar.slider('Promedio (0-10)', min_value=0.0, max_value=10.0, value=7.5, step=0.1)
asistencia = st.sidebar.slider('Asistencia (%)', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
horas_estudio = st.sidebar.slider('Horas de Estudio Semanales', min_value=0, max_value=50, value=15, step=1)
uso_plataforma = st.sidebar.slider('Horas de Uso de Plataforma Semanales', min_value=0.0, max_value=40.0, value=10.0, step=0.5)
materias_perdidas = st.sidebar.slider('Materias Perdidas', min_value=0, max_value=5, value=0, step=1)
nivel_socioeconomico = st.sidebar.selectbox('Nivel Socioeconómico', options=[0, 1, 2], format_func=lambda x: {0: 'Bajo', 1: 'Medio', 2: 'Alto'}[x])
trabaja = st.sidebar.selectbox('¿Trabaja?', options=[0, 1], format_func=lambda x: {0: 'No', 1: 'Sí'}[x])
acceso_internet = st.sidebar.selectbox('¿Acceso a Internet?', options=[0, 1], format_func=lambda x: {0: 'No', 1: 'Sí'}[x])

# Crear un DataFrame con los datos de entrada
input_data = pd.DataFrame([
    {
        'edad': edad,
        'promedio': promedio,
        'asistencia': asistencia,
        'horas_estudio': horas_estudio,
        'uso_plataforma': uso_plataforma,
        'materias_perdidas': materias_perdidas,
        'nivel_socioeconomico': nivel_socioeconomico,
        'trabaja': trabaja,
        'acceso_internet': acceso_internet
    }
])

st.subheader('Datos del Estudiante Ingresados:')
st.write(input_data)

# Botón de predicción
if st.button('Predecir Deserción'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] # Probabilidad de deserción (clase 1)

    st.subheader('Resultado de la Predicción:')
    if prediction == 1:
        st.error(f"❌ ¡ALERTA! El estudiante tiene una alta probabilidad de desertar ({prediction_proba:.2f}).")
        st.write("**Recomendación:** Considerar intervenciones académicas y de apoyo tempranas.")
    else:
        st.success(f"✅ Es poco probable que el estudiante deserte ({prediction_proba:.2f}).")
        st.write("**Recomendación:** Continuar monitoreando su progreso.")

    st.write(f"Probabilidad de Deserción: {prediction_proba:.2f}")

# Footer
st.markdown("---")
st.markdown("Aplicación desarrollada para la predicción de deserción estudiantil.")
