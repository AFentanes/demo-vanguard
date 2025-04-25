import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import time

# --- Carga de datos ---
df = pd.read_csv('demo_productividad_maquinas.csv')

# --- Encabezado del dashboard ---
st.set_page_config(page_title="Dashboard de Productividad", layout="wide")
st.title("🔮 Simulador de Productividad por IA")
st.caption("Predicción semanal para máquinas circulares")
st.markdown("---")

# --- Filtros ---
semana_seleccionada = st.selectbox("Selecciona una semana para predecir", sorted(df['Semana'].unique()))
maquina_seleccionada = st.selectbox("Selecciona la máquina", sorted(df['Máquina'].unique()))

# --- Preparar los datos ---
campos_modelo = [
    'DISPAROS', 'HOYOS', 'MALLAS',
    'MINUTOS RUTINAS DE MANTENIMIENTO',
    'MINUTOS DE MANTENIMIENTO CORRECTIVO DE OPERADOR',
    'MINUTOS DE MANTENIMIENTO CORRECTIVO DE MÁQUINA',
    'MINUTOS DE LIMPIEZA DE MALLAS',
    'MINUTOS CAÍDA DE TELA CARDIGAN POR MÁQUINA'
]

df_modelo = df[df['Máquina'] == maquina_seleccionada]
X = df_modelo[campos_modelo]
y = df_modelo['ProductividadReal']

# --- Entrenamiento del modelo ---
modelo = LinearRegression()
modelo.fit(X, y)

# --- Predicción de la semana seleccionada ---
df_pred = df_modelo[df_modelo['Semana'] == semana_seleccionada]
X_pred = df_pred[campos_modelo]

# Spinner para mostrar que el modelo está trabajando
with st.spinner('🤖 Generando predicción con IA...'):
    time.sleep(1.5)
    if not X_pred.empty:
        prediccion = modelo.predict(X_pred).mean()
        st.success(f"📈 Predicción de productividad para la semana {semana_seleccionada}: **{prediccion:.2f}** unidades")
    else:
        st.warning("No hay datos para la semana seleccionada.")
        prediccion = None

st.markdown("---")

# --- Gráfica de comparación ---
if prediccion:
    fig, ax = plt.subplots()
    ax.plot(df_modelo['Semana'], df_modelo['ProductividadReal'], marker='o', label='Real')
    ax.axhline(y=prediccion, color='orange', linestyle='--', label='Predicción IA')
    ax.set_xlabel("Semana")
    ax.set_ylabel("Productividad")
    ax.set_title(f"📊 Comparativa de Productividad - Máquina {maquina_seleccionada}")
    ax.legend()
    st.pyplot(fig)

# --- Importancia de variables ---
st.markdown("### 🧠 Variables que más afectan la predicción")
importancias = permutation_importance(modelo, X, y, n_repeats=5, random_state=42)
importancia_df = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importancias.importances_mean
}).sort_values(by='Importancia', ascending=False)

st.bar_chart(importancia_df.set_index('Variable'))

# --- Footer ---
st.markdown("---")
st.caption("Desarrollado por Inteligenccia © 2025")
