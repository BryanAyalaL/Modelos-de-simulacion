# Imports necesarios
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

# Cargamos los datos de entrada
data = pd.read_csv("./articulos_ml.csv")

# Veamos cuántas dimensiones y registros contiene el DataFrame
print(data.shape)

# Para ver todas las columnas
pd.set_option('display.max_columns', None)

# Ver las primeras 5 filas en formato tabla
print(data.head())

# Ahora veamos algunas estadísticas de nuestros datos
print(data.describe())

# Visualizamos rápidamente las características de entrada (eliminando algunas columnas)
data.drop(['Title', 'url', 'Elapsed days'], axis=1).hist()
plt.show()

# Vamos a RECORTAR los datos en la zona donde se concentran más los puntos
# esto es en el eje X: entre 0 y 3.500
# y en el eje Y: entre 0 y 80.000
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

colores = ['orange', 'blue']
tamanios = [30, 60]

f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de Cantidad de Palabras
asignar = []
for index, row in filtered_data.iterrows():
    if row['Word count'] > 1808:
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])

# Graficamos los puntos de dispersión
plt.scatter(f1, f2, c=asignar, s=tamanios[0])

# Creamos el objeto de Regresión Lineal
regr = linear_model.LinearRegression()

# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
X_train = np.array(filtered_data[["Word count"]])
y_train = filtered_data['# Shares'].values

# Entrenamos el modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones
y_pred = regr.predict(X_train)

# Creamos un rango de valores para predecir la línea de regresión
x_range = np.linspace(min(f1), max(f1), 100).reshape(-1, 1)
y_range = regr.predict(x_range)

# Graficamos la línea de regresión sobre la dispersión
plt.plot(x_range, y_range, color='red', label='Línea de Regresión')

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

# Veamos los coeficientes obtenidos, en nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))

# Vamos a comprobar:
# Quiero predecir cuántos "Shares" voy a obtener por un artículo con 2.000 palabras,
# según nuestro modelo, hacemos:
y_Dosmil = regr.predict([[2000]])
print(int(y_Dosmil))


# Vamos a intentar mejorar el Modelo, con una dimensión más: 
# Para poder graficar en 3D, haremos una variable nueva que será la suma de los enlaces, comentarios e imágenes
suma = (filtered_data["# of Links"] + filtered_data['# of comments'].fillna(0) + filtered_data['# Images video'])
 
dataX2 =  pd.DataFrame()
dataX2["Word count"] = filtered_data["Word count"]
dataX2["suma"] = suma
XY_train = np.array(dataX2)
z_train = filtered_data['# Shares'].values

# Creamos un nuevo objeto de Regresión Lineal
regr2 = linear_model.LinearRegression()
 
# Entrenamos el modelo, esta vez, con 2 dimensiones
# obtendremos 2 coeficientes, para graficar un plano
regr2.fit(XY_train, z_train)
 
# Hacemos la predicción con la que tendremos puntos sobre el plano hallado
z_pred = regr2.predict(XY_train)
 
# Los coeficientes
print('Coefficients: \n', regr2.coef_)
# Error cuadrático medio
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
# Evaluamos el puntaje de varianza (siendo 1.0 el mejor posible)
print('Variance score: %.2f' % r2_score(z_train, z_pred))

# Creamos una figura para el gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Creamos una malla para graficar el plano de regresión
xx, yy = np.meshgrid(np.linspace(min(filtered_data["Word count"]), max(filtered_data["Word count"]), 30),
                     np.linspace(min(suma), max(suma), 30))

# Calculamos el valor z correspondiente para cada punto en la malla
z = regr2.coef_[0] * xx + regr2.coef_[1] * yy + regr2.intercept_

# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.5, cmap='viridis')

# Graficamos los puntos en 3D (originales)
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue', s=30, label="Puntos originales")

# Graficamos los puntos en 3D (predicciones)
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red', s=40, label="Puntos predichos")

# Configuramos el título y las etiquetas de los ejes
ax.set_xlabel('Word Count')
ax.set_ylabel('Sum of Links, Comments, and Images')
ax.set_zlabel('Shares')

ax.set_title('3D Linear Regression')

# Añadimos leyenda
ax.legend()

# Mostramos el gráfico
plt.show()
