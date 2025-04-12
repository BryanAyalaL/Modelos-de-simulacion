import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de estilo
plt.style.use('ggplot')

# PASO 1: Definir los datos de la imagen
data = {
    "Trimestre": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Reales": [180, 168, 159, 175, 190, 205, 180, 182, np.nan],
    "Pronóstico α=0.10": [np.nan] * 9,
    "Fórmula α=0.10": [""] * 9,
    "Pronóstico α=0.50": [np.nan] * 9,
    "Fórmula α=0.50": [""] * 9
}

df = pd.DataFrame(data).set_index("Trimestre")

# Inicializar pronósticos manualmente (según la imagen)
df.loc[1, "Pronóstico α=0.10"] = 175.0
df.loc[1, "Pronóstico α=0.50"] = 175.0

# PASO 2: Calcular pronósticos y fórmulas
for i in range(2, 10):
    # Suavizamiento α=0.10
    real_anterior = df.loc[i-1, "Reales"]
    pron_anterior_01 = df.loc[i-1, "Pronóstico α=0.10"]
    pron_actual_01 = pron_anterior_01 + 0.10 * (real_anterior - pron_anterior_01)
    df.loc[i, "Pronóstico α=0.10"] = pron_actual_01
    df.loc[i, "Fórmula α=0.10"] = f"{pron_anterior_01:.2f} + 0.10({real_anterior} - {pron_anterior_01:.2f})"
    
    # Suavizamiento α=0.50
    pron_anterior_05 = df.loc[i-1, "Pronóstico α=0.50"]
    pron_actual_05 = pron_anterior_05 + 0.50 * (real_anterior - pron_anterior_05)
    df.loc[i, "Pronóstico α=0.50"] = pron_actual_05
    df.loc[i, "Fórmula α=0.50"] = f"{pron_anterior_05:.2f} + 0.50({real_anterior} - {pron_anterior_05:.2f})"

# PASO 3: Imprimir tabla formateada
try:
    from tabulate import tabulate
    print("Tabla de Pronósticos:")
    print(tabulate(
        df.reset_index(),
        headers=["Trimestre", "Reales", "Pronóstico α=0.10", "Fórmula α=0.10", "Pronóstico α=0.50", "Fórmula α=0.50"],
        tablefmt="grid",
        numalign="center",
        stralign="left",
        missingval="?"
    ))
except ImportError:
    print("Advertencia: Instala 'tabulate' para mejor formato (pip install tabulate)")
    print(df)

# PASO 4: Graficar resultados
plt.figure(figsize=(12, 6))

# Datos reales
plt.plot(df.index, df["Reales"], 'o-', label='Reales', markersize=8, color='#2c3e50')

# Pronósticos α=0.10
plt.plot(df.index, df["Pronóstico α=0.10"], 's--', label='α=0.10', color='#e74c3c')

# Pronósticos α=0.50
plt.plot(df.index, df["Pronóstico α=0.50"], 'D--', label='α=0.50', color='#27ae60')

plt.title('Suavizamiento Exponencial Simple', fontsize=14)
plt.xlabel('Trimestre', fontsize=12)
plt.ylabel('Toneladas Descargadas', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df.index)
plt.tight_layout()
plt.show()