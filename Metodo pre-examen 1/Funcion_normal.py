import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la distribución normal
mu = 0      # Media
sigma = 1   # Desviación estándar

# Rango de valores de x
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

# Función de densidad de probabilidad
def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma * 2))

# Calcular valores de y
y = normal_pdf(x, mu, sigma)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=f'Normal(μ={mu}, σ={sigma})', color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Distribución Normal')
plt.legend()
plt.grid()
plt.show()