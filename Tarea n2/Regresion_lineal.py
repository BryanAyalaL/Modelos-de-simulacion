# Importar bibliotecas principales
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset desde archivo CSV
data = pd.read_csv("./articulos_ml.csv")

# Mostrar información básica del dataset
print("Forma del dataset:", data.shape)

# Configurar visualización de pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Mostrar primeras filas del dataset
print("\nPrimeras 5 filas:")
print(data.head())

# Mostrar estadísticas descriptivas
print("\nEstadísticas básicas:")
print(data.describe())

# Crear histogramas de las características numéricas
data.drop(['Title', 'url', 'Elapsed days'], axis=1).hist()
plt.title('Distribución de características numéricas')
plt.show()

# Filtrar datos para eliminar valores extremos
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

# Configurar colores según cantidad de palabras
colores = ['orange', 'blue']
asignar = [colores[0] if row['Word count'] > 1808 else colores[1] for index, row in filtered_data.iterrows()]

# Crear gráfico de dispersión
plt.scatter(filtered_data['Word count'], filtered_data['# Shares'], c=asignar, s=30)
plt.title('Relación entre palabras y compartimientos')
plt.xlabel('Cantidad de palabras')
plt.ylabel('Veces compartido')

# Entrenar modelo de regresión lineal simple
regr = linear_model.LinearRegression()
X = filtered_data[["Word count"]]
y = filtered_data['# Shares']
regr.fit(X, y)

# Generar línea de regresión
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = regr.predict(x_range)
plt.plot(x_range, y_range, color='red', linewidth=2, label='Predicciones')

plt.legend()
plt.show()

# Mostrar métricas del modelo
print('\nResultados modelo simple:')
print('Pendiente:', regr.coef_[0])
print('Intercepto:', regr.intercept_)
print('Error cuadrático medio:', mean_squared_error(y, regr.predict(X)))
print('Precisión (R²):', r2_score(y, regr.predict(X)))

# Predecir para 2000 palabras
prediccion = regr.predict([[2000]])
print('\nPredicción para 2000 palabras:', int(prediccion[0]))

# Modelo mejorado con segunda característica
dataX2 = filtered_data.assign(
    suma_interacciones = filtered_data["# of Links"] + 
                        filtered_data['# of comments'].fillna(0) + 
                        filtered_data['# Images video']
)[["Word count", "suma_interacciones"]]

# Entrenar modelo de regresión múltiple
regr2 = linear_model.LinearRegression()
regr2.fit(dataX2, y)

# Mostrar métricas del modelo mejorado
print('\nResultados modelo mejorado:')
print('Coeficientes:', regr2.coef_)
print('Error cuadrático medio:', mean_squared_error(y, regr2.predict(dataX2)))
print('Precisión (R²):', r2_score(y, regr2.predict(dataX2)))

# Configurar gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear malla para superficie de predicción
xx, yy = np.meshgrid(np.linspace(dataX2["Word count"].min(), dataX2["Word count"].max(), 20),
                     np.linspace(dataX2["suma_interacciones"].min(), dataX2["suma_interacciones"].max(), 20))
zz = regr2.intercept_ + regr2.coef_[0] * xx + regr2.coef_[1] * yy

# Dibujar superficie de predicción
ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='coolwarm', edgecolor='none')
ax.scatter(dataX2["Word count"], dataX2["suma_interacciones"], y, c='green', s=40, label='Datos reales')
ax.scatter(dataX2["Word count"], dataX2["suma_interacciones"], regr2.predict(dataX2), c='red', s=40, label='Predicciones')

# Personalizar gráfico
ax.set_title('Regresión en 3D: Palabras + Interacciones vs Compartimientos')
ax.set_xlabel('Cantidad de palabras')
ax.set_ylabel('Suma de enlaces/comentarios/imágenes')
ax.set_zlabel('Veces compartido')
ax.view_init(elev=25, azim=45)
ax.legend()

plt.show()

z_Dosmil = regr2.predict([[2000, 10+4+6]])
print(int(z_Dosmil))