import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Demo Productividad Tejido Circular", layout="wide")

st.title("📈 Dashboard de Productividad - Tejido Circular")

# Cargar datos
@st.cache_data

def load_data():
    df = pd.read_csv("demo_productividad_maquinas.csv", parse_dates=['Fecha inicio', 'Fecha fin'])
    return df

df = load_data()

# Procesar columnas
cat_cols = ['Tripulación', 'Máquina', 'Mezcla', 'Proveedor']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Calcular productividad

df['MIN_TOTALES'] = (df['Fecha fin'] - df['Fecha inicio']).dt.total_seconds() / 60

df['MIN_NO_PRODUCTIVOS'] = (
    df['PARO SIN MOTIVO MATRIX'] + df['PAROS SIN MOTIVO MÁQUINA'] +
    df['MIN CAMBIO PLATINAS'] + df['MIN MANTTO RUTINA'] + df['MIN CALIDAD'] +
    df['MIN MANTTO PROGRAMADO'] + df['MIN OTROS PAROS'] +
    df['MIN LIMPIEZA ESTRUCTURA'] + df['MIN LIMPIEZA MALLAS'] +
    df['MIN CORRECTIVO OPERADOR'] + df['MIN CORRECTIVO MÁQUINA'] +
    df['MIN CAIDA CARDIGAN OPERADOR'] + df['MIN CAIDA CARDIGAN MÁQUINA'] +
    df['MIN CAIDA CARDIGAN M. PRIMA'] + df['MIN CAIDA TELA']
)
df['MIN_PRODUCTIVOS'] = df['MIN_TOTALES'] - df['MIN_NO_PRODUCTIVOS']
df['MIN_PRODUCTIVOS'] = df['MIN_PRODUCTIVOS'].apply(lambda x: max(x, 1))
df['Productividad'] = (df['Fin vueltas'] - df['Inicio vueltas']) / df['MIN_PRODUCTIVOS']

# Modelo de predicción
features = [
    'Tripulación', 'Semana', 'Máquina', 'Calibre', 'Mezcla', 'Proveedor',
    'DISPAROS', 'HOYOS', 'MALLAS',
    'PARO SIN MOTIVO MATRIX', 'PAROS SIN MOTIVO MÁQUINA', 'CAMBIO DE AGUJAS',
    'MIN CAMBIO PLATINAS', 'MIN MANTTO RUTINA', 'MIN CALIDAD', 'MIN MANTTO PROGRAMADO',
    'MIN OTROS PAROS', 'MIN LIMPIEZA ESTRUCTURA', 'MIN LIMPIEZA MALLAS',
    'MIN CORRECTIVO OPERADOR', 'MIN CORRECTIVO MÁQUINA',
    'MIN CAIDA CARDIGAN OPERADOR', 'MIN CAIDA CARDIGAN MÁQUINA',
    'MIN CAIDA CARDIGAN M. PRIMA', 'MIN CAIDA TELA'
]

X = df[features]
y = df['Productividad']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicción'] = model.predict(X)

# Filtros
with st.sidebar:
    semana_sel = st.multiselect("Filtrar por semana:", options=sorted(df['Semana'].unique()), default=sorted(df['Semana'].unique()))
    maquina_sel = st.multiselect("Filtrar por máquina:", options=sorted(df['Máquina'].unique()), default=sorted(df['Máquina'].unique()))

filtro = df[df['Semana'].isin(semana_sel) & df['Máquina'].isin(maquina_sel)]

# Gráficos
st.subheader("📊 Productividad real vs estimada")
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=filtro, x='Semana', y='Productividad', label='Real', ax=ax)
sns.lineplot(data=filtro, x='Semana', y='Predicción', label='Predicción', ax=ax)
plt.ylabel("Vueltas por minuto útil")
st.pyplot(fig)

st.subheader("🏆 Ranking de productividad por máquina")
ranking = filtro.groupby('Máquina')['Productividad'].mean().sort_values(ascending=False)
st.bar_chart(ranking)

st.subheader("📈 Importancia de características")
importances = model.feature_importances_
importancia_df = pd.Series(importances, index=features).sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(10, 8))
importancia_df.tail(10).plot(kind='barh', ax=ax2)
st.pyplot(fig2)

st.markdown("---")
st.markdown("Desarrollado para demo con datos sintéticos inteligenccia.com - Tejido Circular")