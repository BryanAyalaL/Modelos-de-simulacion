# sea n1=a*n(i+1)+b    m=100
def congruenciales(seed, n, A=13, B=5, M=100):
    """
    Genera una secuencia de números pseudoaleatorios usando el método congruencial lineal.
    
    :param seed: Número inicial o semilla
    :param n: Cantidad de números a generar
    :param A: Multiplicador
    :param B: Incremento
    :param M: Módulo
    :return: Lista con la secuencia de números generados
    """
    results = []
    current = seed
    
    for _ in range(n):
        # Aplicar la fórmula congruencial
        current = (A * current + B) % M
        results.append(current / M)  # Normalizar el número entre 0 y 1
    
    return results

# Ejemplo de uso
seed = 7  # Semilla inicial
n = 50  # Número de valores aleatorios a generar
random_numbers = congruenciales(seed, n)
print(random_numbers)

import numpy as np
print(np.mean(random_numbers))