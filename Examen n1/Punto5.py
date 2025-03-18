import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

# 1. Datos
df = pd.read_csv("Examen n1/Tasa_de_inter_s_bancario_corriente_-TIBC_20250318.csv")
date_cols = ['FECHA_RESOLUCION', 'VIGENCIA_DESDE', 'VIGENCIA_HASTA']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

df['AÑO_RESOLUCION'] = df['FECHA_RESOLUCION'].dt.year
df['DURACION_VIGENCIA'] = (df['VIGENCIA_HASTA'] - df['VIGENCIA_DESDE']).dt.days

df = pd.get_dummies(df, columns=['MODALIDAD'], prefix='MODALIDAD', drop_first=True)
df = df.dropna(subset=['INTERES_BANCARIO_CORRIENTE'])
df = df[df['DURACION_VIGENCIA'] > 0]

scaler = StandardScaler()
numeric_features = ['AÑO_RESOLUCION', 'DURACION_VIGENCIA']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 2. Modelos
X_simple = df[['AÑO_RESOLUCION']]
y = df['INTERES_BANCARIO_CORRIENTE']
modelo_simple = LinearRegression().fit(X_simple, y)

X_2 = df[numeric_features + list(df.filter(like='MODALIDAD').columns)]
modelo2 = LinearRegression().fit(X_2, y)

# 3. Gráficas

# Gráfico 1: Tendencia
plt.figure(figsize=(10, 6))
plt.scatter(df['AÑO_RESOLUCION'], y, alpha=0.5, color='blue', label='Datos')
plt.plot(df['AÑO_RESOLUCION'], modelo_simple.predict(X_simple), color='red',
         linewidth=2, label=f'Tendencia (R² = {modelo_simple.score(X_simple, y):.2f})')
plt.title('Relación Año vs Tasa de Interés')
plt.xlabel('Año (Normalizado)')
plt.ylabel('Tasa de Interés (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico 2: Residuos
plt.figure(figsize=(10, 6))
residuos = y - modelo2.predict(X_2)
predichos = modelo2.predict(X_2)
plt.scatter(predichos, residuos, alpha=0.6, color='green')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Análisis de Residuos')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico 3: Tasas por Modalidad
plt.figure(figsize=(14, 8))
modalidades = [col for col in df.columns if 'MODALIDAD' in col]
colores = plt.cm.tab20(np.linspace(0, 1, len(modalidades)))
for i, modalidad in enumerate(modalidades):
    subset = df[df[modalidad] == 1]
    if not subset.empty:
        plt.scatter(subset['AÑO_RESOLUCION'], subset['INTERES_BANCARIO_CORRIENTE'],
                    color=colores[i], label=modalidad.replace('MODALIDAD_', ''),
                    alpha=0.7, s=80)
plt.title('Distribución de Tasas por Modalidad', fontsize=14)
plt.xlabel('Año (Normalizado)', fontsize=12)
plt.ylabel('Tasa de Interés (%)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Resultados
coef = pd.DataFrame({
    'Variable': X_2.columns,
    'Coeficiente': modelo2.coef_,
    'Impacto (%)': np.abs(modelo2.coef_)*100
}).sort_values('Impacto (%)', ascending=False)

print("\n" + "="*60)
print("Coeficientes Estandarizados")
print("="*60)
print(coef)
print(f"\nR² del modelo: {modelo2.score(X_2, y):.4f}")
