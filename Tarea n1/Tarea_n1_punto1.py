def mid_square_random(seed, n, digits=2):
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
if __name__ == '__main__':
    try:
        seed = int(input("Ingrese la semilla inicial (número entero): "))
        n = int(input("Ingrese la cantidad de números a generar: "))
    except ValueError:
        print("Error: Ingrese valores numéricos válidos.")
        exit(1) # Finaliza e indica que fue por un error 
    
    random_numbers = mid_square_random(seed, n)
    
    print("\nSecuencia de números pseudoaleatorios generada:")
    print(random_numbers)
    
    import numpy as np
    print("\nMedia de los números generados:", np.mean(random_numbers))
