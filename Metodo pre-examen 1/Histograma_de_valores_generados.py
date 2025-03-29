import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros de la distribución normal
mu = 5       # Media
sigma = 1    # Desviación estándar

# Función para generar un número aleatorio con distribución normal
def generar_normal(mu, sigma):
    return np.random.normal(mu, sigma)

# Generar 1000 números aleatorios con distribución normal
numeros_aleatorios = [generar_normal(mu, sigma) for _ in range(1000)]

# Calcular la media muestral
media_muestral = np.mean(numeros_aleatorios)

# Calcular la desviación estándar poblacional
desviacion_estandar_poblacional = np.std(numeros_aleatorios, ddof=0)

# Calcular la desviación estándar muestral
desviacion_estandar_muestral = np.std(numeros_aleatorios, ddof=1)

# Imprimir valores calculados
print(f"Media muestral: {media_muestral}")
print(f"Desviación estándar poblacional: {desviacion_estandar_poblacional}")
print(f"Desviación estándar muestral: {desviacion_estandar_muestral}")

# Crear histograma
plt.figure(figsize=(8, 5))
plt.hist(numeros_aleatorios, bins=30, density=True, alpha=0.6, color='blue', label='Histograma')

# Rango de valores para la función de densidad
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm.pdf(x, mu, sigma)

# Graficar la función de densidad teórica
plt.plot(x, y, 'r-', label=f'Distribución Normal (μ={mu}, σ={sigma})')

# Configuración del gráfico
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.title('Histograma de valores generados y distribución teórica')
plt.legend()
plt.grid()
plt.show()
