import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Datos de ejemplo: asignación de recursos (X) y rendimiento del proceso (Y)
asignacion_recursos = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
rendimiento_proceso = np.array([15, 25, 35, 45, 55])

# Crear un modelo de regresión lineal
modelo_regresion = LinearRegression()

# Entrenar el modelo con los datos
modelo_regresion.fit(asignacion_recursos, rendimiento_proceso)

# Hacer predicciones para diferentes asignaciones de recursos
asignacion_recursos_prediccion = np.array([15, 25, 35, 45, 55]).reshape(-1, 1)
rendimiento_prediccion = modelo_regresion.predict(asignacion_recursos_prediccion)

# Imprimir los coeficientes de la regresión
interseccion = modelo_regresion.intercept_
pendiente = modelo_regresion.coef_[0]
print("Intersección (a):", interseccion)
print("Pendiente (b):", pendiente)

# Obtener y mostrar el R^2
r2 = modelo_regresion.score(asignacion_recursos, rendimiento_proceso)
print("Coeficiente de determinación R^2:", r2)

# Graficar los datos y la línea de regresión
plt.scatter(asignacion_recursos, rendimiento_proceso, color='blue', label='Datos')
plt.plot(asignacion_recursos_prediccion, rendimiento_prediccion, color='red', label='Regresión Lineal')
plt.xlabel('Asignación de Recursos')
plt.ylabel('Rendimiento del Proceso')
plt.title('Regresión Lineal: Asignación de Recursos vs. Rendimiento del Proceso')
plt.legend()
plt.show()
