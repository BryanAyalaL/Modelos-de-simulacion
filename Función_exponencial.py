import numpy as np
import matplotlib.pyplot as plt

# Parámetro lambda de la distribución exponencial
lmbda = 1.5  # Puedes cambiar este valor

# Rango de valores de x (solo valores positivos)
x = np.linspace(0, 5, 1000)

# Función de densidad de probabilidad de la distribución exponencial
def exponencial_pdf(x, lmbda):
    return lmbda * np.exp(-lmbda * x)

# Calcular valores de y
y = exponencial_pdf(x, lmbda)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=f'Exponencial(λ={lmbda})', color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Distribución Exponencial')
plt.legend()
plt.grid()
plt.show()