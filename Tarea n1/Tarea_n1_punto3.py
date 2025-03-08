import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    # Parámetro lambda de la distribución de Poisson
    lmbda = 3
    size = 10  # Cantidad de números a generar
    
    # Generar 10 números aleatorios con distribución de Poisson
    poisson_numbers = np.random.poisson(lam=lmbda, size=size)
    
    # Mostrar los números generados en la consola
    print("Números aleatorios de distribución de Poisson (λ = 3):")
    print(poisson_numbers)
    
    # Configurar la figura para el histograma
    plt.figure(figsize=(8, 6))
    
    # Definir los bordes de los bins para representar datos discretos adecuadamente
    bins = np.arange(min(poisson_numbers), max(poisson_numbers) + 2) - 0.5
    
    # Graficar el histograma de los datos generados
    plt.hist(poisson_numbers, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black', label='Datos generados')
    
    # Calcular la frecuencia teórica esperada usando la PMF de Poisson
    # Se consideran todos los valores enteros entre el mínimo y el máximo de los datos generados.
    k_values = np.arange(int(np.floor(min(poisson_numbers))), int(np.ceil(max(poisson_numbers))) + 1)
    # Para cada k se calcula P(X=k) y se escala por el tamaño de la muestra (size)
    expected_counts = size * (np.exp(-lmbda) * (lmbda ** k_values) / np.array([math.factorial(k) for k in k_values]))
    
    # Agregar la línea de la frecuencia teórica
    plt.plot(k_values, expected_counts, marker='o', color='red', linestyle='-', label='Valor teórico')
    
    # Configuración de la gráfica
    plt.title('Histograma de la Distribución de Poisson (λ = 3)')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
