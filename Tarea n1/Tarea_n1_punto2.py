import numpy as np
import matplotlib.pyplot as plt

def main():
    lam = 0.5
    scale = 1 / lam  # scale es el inverso de lambda
    
    # Generar 10 números aleatorios con distribución exponencial
    random_numbers = np.random.exponential(scale=scale, size=10)
    
    print("Números aleatorios con distribución exponencial (lambda = 0.5):")
    for num in random_numbers:
        print(num)
    
    # Crear un histograma para visualizar los números generados
    plt.figure(figsize=(8, 6))
    plt.hist(random_numbers, bins=5, color='skyblue', edgecolor='black')
    plt.title('Histograma de Números Aleatorios con Distribución Exponencial')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
