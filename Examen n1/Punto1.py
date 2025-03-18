'''Escribe un programa en Python que genere 1200 números aleatorios con 
una  distribución  exponencial  con  parámetro  lambda  igual  a  0.8  utilizando  la 
biblioteca numpy. Genera el respectivo histograma y comprueba que los datos 
cumplen con una distribución exponencial, explicando tu respuesta.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

lambda_param = 0.8
muestra = 1200

# Generar datos exponenciales
np.random.seed(42)  # Para reproducibilidad
datos = np.random.exponential(scale=1/lambda_param, size=muestra)

# Crear histograma
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(datos, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')

# Superponer la PDF teórica
x = np.linspace(0, np.max(datos), 1000)
pdf_teorica = lambda_param * np.exp(-lambda_param * x)
plt.plot(x, pdf_teorica, 'r-', lw=2, label='PDF Teórica')

plt.title('Histograma vs Distribución Exponencial Teórica')
plt.xlabel('Valor')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid(True)
plt.show()

# Prueba de Kolmogorov-Smirnov
ks_stat, p_valor = stats.kstest(datos, 'expon', args=(0, 1/lambda_param))  # args=(loc, scale)

print("\nResultado de la prueba KS:")
print(f"Estadístico KS: {ks_stat:.4f}")
print(f"Valor p: {p_valor:.4f}")

# Interpretación
# estandar es de α = 0.05
alpha = 0.05
if p_valor > alpha:
    print("\nConclusión: Los datos siguen una distribución exponencial (no se rechaza H0)")
else:
    print("\nConclusión: Los datos NO siguen una distribución exponencial (se rechaza H0)")