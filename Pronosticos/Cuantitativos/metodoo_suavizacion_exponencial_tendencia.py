import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del modelo
alpha = 0.3
beta = 0.4

# Datos de demanda
Y = [74, 79, 80, 90, 105, 142, 122]
n = len(Y)

# Inicializar listas para pronósticos, tendencia y FIT
F = [0] * (n + 1)  # Pronósticos (F_{t+1})
T = [0] * (n + 1)   # Tendencia (T_{t+1})
FIT = [0] * (n + 1) # FIT_{t}

# Valores iniciales (t=1)
F[0] = Y[0]  # F1 = Y1
T[0] = 0      # T1 = 0
FIT[0] = F[0] + T[0]  # FIT1 = F1 + T1

# Calcular pronósticos, tendencia y FIT para cada periodo
for i in range(1, n + 1):
    # Obtener demanda del periodo anterior
    Y_prev = Y[i-1] if i <= n else None

    # Calcular pronóstico F_{i+1}
    FIT_prev = FIT[i-1]
    F[i] = FIT_prev + alpha * (Y_prev - FIT_prev)

    # Calcular tendencia T_{i+1}
    T_prev = T[i-1]
    T[i] = T_prev + beta * (F[i] - FIT_prev)

    # Calcular FIT_{i+1}
    FIT[i] = F[i] + T[i]

# Crear DataFrame para mostrar la tabla
data = []
for t in range(1, n + 2):
    y_t = Y[t-1] if t <= n else None
    f_next = round(F[t], 3) if t <= n else None
    t_next = round(T[t], 3) if t <= n else None
    fit_current = round(FIT[t-1], 3) if t <= n + 1 else None

    data.append({
        'Tiempo (t)': t,
        'Demanda (Y_t)': y_t,
        'Pronóstico (F_{t+1})': f_next,
        'Tendencia (T_{t+1})': t_next,
        'FIT_t': fit_current
    })

df = pd.DataFrame(data)

# Mostrar tabla
print("Tabla de Pronósticos con Suavizamiento Exponencial y Tendencia:")
print(df.to_string(index=False))

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(df['Tiempo (t)'][:-1], df['Demanda (Y_t)'][:-1], marker='o', label='Demanda Real')
plt.plot(df['Tiempo (t)'], df['Pronóstico (F_{t+1})'], marker='s', linestyle='--', label='Pronóstico (F)')
plt.xlabel('Periodo (t)')
plt.ylabel('Valor')
plt.title('Suavizamiento Exponencial con Tendencia ')
plt.legend()
plt.grid(True)
plt.show()