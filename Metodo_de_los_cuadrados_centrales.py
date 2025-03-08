def mid_square_random(seed, n, digits=2):
    """
    Genera una secuencia de números pseudoaleatorios usando el método de cuadrados centrales aleatorios.
    
    :param seed: Número inicial o semilla
    :param n: Cantidad de números a generar
    :param digits: Número de dígitos a considerar en la semilla
    :return: Lista con la secuencia de números generados
    """
    results = []
    
    for _ in range(n):
        # Elevar la semilla al cuadrado
        seed = seed ** 2
        
        # Convertir el número al cuadrado en una cadena
        str_seed = str(seed)
        
        # Si el número tiene más de 3 dígitos, tomamos los dígitos centrales
        if len(str_seed) > 3:
            mid = len(str_seed) // 2
            seed = int(str_seed[mid - digits // 2 : mid + digits // 2])  # Tomamos el centro
        else:
            # Si tiene 3 dígitos, tomamos los dos últimos
            seed = int(str_seed[-2:])
        
        # Normalizamos el número a un rango entre 0 y 1
        results.append(seed / 100)  # Normalizar entre 0 y 1

    return results

# Ejemplo de uso
seed = 73  # Semilla inicial
n = 25  # Número de valores aleatorios a generar
random_numbers = mid_square_random(seed, n)
print(random_numbers)


import numpy as np
print(np.mean(random_numbers))